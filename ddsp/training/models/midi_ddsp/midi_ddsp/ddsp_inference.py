"""Model class for DDSP Inference module used in MIDI-DDSP."""

import tensorflow as tf
import ddsp
from utils.audio_io import tf_log_mel
from data_handling.instrument_name_utils import NUM_INST
import ddsp.training.nn as nn
from ddsp.spectral_ops import F0_RANGE, LD_RANGE

tfk = tf.keras
tfkl = tfk.layers


class MelF0LDEncoder(tfkl.Layer):
  def __init__(self, cnn, nhid, sample_rate, win_length, hop_length, n_fft,
               num_mels, fmin):
    """The encoder in DDSP Inference.
    The MelF0LDEncoder takes input of audio, loudness and f0.
    The MelF0LDEncoder extract features from audio using an 8-layer CNN,
    and extract features from loudness and f0 using fully-connected layers.
    Then, a bi-lstm is used to extract contextual features from the extracted
    features.
    """
    super(MelF0LDEncoder, self).__init__()
    self.nhid = nhid
    self.cnn = cnn
    self.z_fc = tfkl.Dense(nhid)
    self.f0_ld_fc = tfkl.Dense(nhid)
    self.rnn = tfkl.Bidirectional(
      tfkl.LSTM(
        units=nhid, return_sequences=True,
      ),
      name="bilstm",
    )
    # TODO(yusongwu): change emb dim to 64
    self.instrument_emb = tfkl.Embedding(NUM_INST, 128)

    # mel-spec parameters
    self.sample_rate = sample_rate
    self.win_length = win_length
    self.hop_length = hop_length
    self.n_fft = n_fft
    self.num_mels = num_mels
    self.fmin = fmin

  def call(self, inputs, training=False, mask=None):
    mel = tf_log_mel(inputs['audio'],
                     self.sample_rate,
                     self.win_length,
                     self.hop_length,
                     self.n_fft,
                     self.num_mels,
                     self.fmin)
    z_cnn = self.cnn(mel, training=training)
    z_reduce = self.z_fc(z_cnn)
    instrument_z = tf.tile(
      self.instrument_emb(inputs['instrument_id'])[:, tf.newaxis, :],
      [1, z_cnn.shape[1], 1])
    x = tf.concat([ddsp.core.hz_to_midi(inputs['f0_hz']) / F0_RANGE,
                   inputs['loudness_db'] / LD_RANGE], axis=-1)
    x_z = self.f0_ld_fc(x)
    z_out = self.rnn(tf.concat([x_z, z_reduce, instrument_z], axis=-1))
    return z_out


class FCHarmonicDecoder(tfkl.Layer):
  """The decoder in DDSP Inference.
  The FCHarmonicDecoder takes input of a feature sequence,
  and output the synthesis parameters for DDSP through fully-connected layers.
  """

  def __init__(self, nhramonic=100, nnoise=65):
    super(FCHarmonicDecoder, self).__init__()

    self.harmonic_amp_fc = tfkl.Dense(1, bias_initializer='ones')
    self.harmonic_distribution_fc = tfkl.Dense(nhramonic)

    self.noise_mag_fc = tfkl.Dense(nnoise)

  def get_synth_params(self, inputs):
    z, data = inputs

    harmonic_amp = self.harmonic_amp_fc(z)
    harmonic_distribution = self.harmonic_distribution_fc(z)
    noise_mag = self.noise_mag_fc(z)

    synth_params = {
      'f0_hz': data['f0_hz'],
      'amplitudes': harmonic_amp,
      'harmonic_distribution': harmonic_distribution,
      'noise_magnitudes': noise_mag,
    }

    return synth_params

  def call(self, inputs, training=None, mask=None, use_f0_gt=False,
           f0_pred_input=None):
    synth_params = self.get_synth_params(inputs)

    return synth_params


class F0LDEncoder(tfkl.Layer):
  """The encoder of original DDSP autoencoder."""

  # TODO: (yusongwu) To be removed and use the decoders.RnnFcDecoder
  def __init__(self):
    super(F0LDEncoder, self).__init__()
    self.nhid = 512
    self.f0_fc = nn.FcStack(self.nhid, layers=3)
    self.ld_fc = nn.FcStack(self.nhid, layers=3)
    self.instrument_emb = tfkl.Embedding(NUM_INST, 128)
    self.rnn = tfkl.GRU(
      units=self.nhid, return_sequences=True,  # dropout=0.2,
    )

  def call(self, inputs, training=False, mask=None):
    z_f0 = self.f0_fc(ddsp.core.hz_to_midi(inputs['f0_hz']) / F0_RANGE)
    z_ld = self.ld_fc(inputs['loudness_db'] / LD_RANGE)
    instrument_z = tf.tile(
      self.instrument_emb(inputs['instrument_id'])[:, tf.newaxis, :],
      [1, z_ld.shape[1], 1])
    x_z = tf.concat([z_f0, z_ld, instrument_z], axis=-1)
    z_out = self.rnn(x_z)
    z_out = tf.concat([x_z, z_out], axis=-1)
    return z_out


class FCStackHarmonicDecoder(tfkl.Layer):
  """The decoder original DDSP autoencoder.
  The FCStackHarmonicDecoder takes input of a feature sequence,
  and output the synthesis parameters for DDSP through stacked MLP.
  """
  def __init__(self, nharmonic=100, nnoise=65):
    super(FCStackHarmonicDecoder, self).__init__()

    self.output_splits = (
      ('amplitudes', 1), ('harmonic_distribution', nharmonic),
      ('noise_magnitudes', nnoise))
    self.n_out = sum([v[1] for v in self.output_splits])
    self.out_stack = nn.FcStack(512, layers=3)
    self.dense_out = tfkl.Dense(self.n_out)

  def get_synth_params(self, inputs):
    z, data = inputs

    z_output = self.out_stack(z)
    synth_params = nn.split_to_dict(self.dense_out(z_output),
                                    self.output_splits)

    synth_params['f0_hz'] = data['f0_hz']

    return synth_params

  def call(self, inputs, training=None, mask=None, use_f0_gt=False,
           f0_pred_input=None):
    synth_params = self.get_synth_params(inputs)

    return synth_params


class ConvBlock(tfkl.Layer):
  def __init__(self, out_channels, pool_size=(2, 2)):
    """
    A tensorflow implementation of ConvBlock used in audioset classification.
    This CNN has better performance when used in spectrogram feature
    extraction for audio tagging. Adapted from pytorch implementation:
    https://github.com/qiuqiangkong/audioset_tagging_cnn.
    paper: https://arxiv.org/abs/1912.10211.
    Args:
      out_channels: number of output channels.
      pool_size: size of pooling, in height and width.
    """

    super(ConvBlock, self).__init__()

    self.conv1 = tfkl.Conv2D(filters=out_channels,
                             kernel_size=(3, 3), strides=(1, 1),
                             padding='same', use_bias=False,
                             kernel_initializer=tf.keras.initializers.GlorotUniform())

    self.conv2 = tfkl.Conv2D(filters=out_channels,
                             kernel_size=(3, 3), strides=(1, 1),
                             padding='same', use_bias=False,
                             kernel_initializer=tf.keras.initializers.GlorotUniform())

    self.bn1 = tfkl.BatchNormalization(beta_initializer='zeros',
                                                gamma_initializer='ones')
    self.bn2 = tfkl.BatchNormalization(beta_initializer='zeros',
                                                gamma_initializer='ones')

    self.max_pool = tfkl.MaxPool2D(pool_size=pool_size, padding='same')
    self.avg_pool = tfkl.AveragePooling2D(pool_size=pool_size, padding='same')

  def call(self, inputs, training=None, pool_type='avg'):
    x = inputs
    x = tf.nn.relu(self.bn1(self.conv1(x), training=training))
    x = tf.nn.relu(self.bn2(self.conv2(x), training=training))
    if pool_type == 'max':
      x = self.max_pool(x)
    elif pool_type == 'avg':
      x = self.avg_pool(x)
    elif pool_type == 'avg+max':
      x1 = self.avg_pool(x)
      x2 = self.max_pool(x)
      x = x1 + x2
    else:
      raise Exception('Incorrect argument!')

    return x


class Cnn8(tfkl.Layer):
  """
  A tensorflow implementation of CNN8 used in audioset classification.
  This CNN has better performance when used in spectrogram feature
  extraction for audio tagging. Adapted from pytorch implementation:
  https://github.com/qiuqiangkong/audioset_tagging_cnn.
  paper: https://arxiv.org/abs/1912.10211.
  """

  def __init__(self, pool_size=(2, 2), dropout=0.2):
    super(Cnn8, self).__init__()

    self.conv_block1 = ConvBlock(out_channels=64, pool_size=pool_size)
    self.conv_block2 = ConvBlock(out_channels=128, pool_size=pool_size)
    self.conv_block3 = ConvBlock(out_channels=256, pool_size=pool_size)
    self.conv_block4 = ConvBlock(out_channels=512, pool_size=pool_size)

    self.dropout = tfkl.Dropout(rate=dropout)

  def call(self, x, training=None):
    x = x[..., tf.newaxis]

    x = self.conv_block1(x, pool_type='avg', training=training)
    x = self.dropout(x, training=training)
    x = self.conv_block2(x, pool_type='avg', training=training)
    x = self.dropout(x, training=training)
    x = self.conv_block3(x, pool_type='avg', training=training)
    x = self.dropout(x, training=training)
    x = self.conv_block4(x, pool_type='avg', training=training)
    x = self.dropout(x, training=training)
    x = tf.reshape(x, [x.shape[0], x.shape[1], -1])

    return x
