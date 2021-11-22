
from tqdm import tqdm
import tensorflow as tf
import ddsp
import ddsp.training.nn as nn
from midi_ddsp.cond_rnn import TwoLayerCondAutoregRNN

tfk = tf.keras
tfkl = tfk.layers

# Quantization constants for f0 and loudness.

# [-1, 1] in cents (0.01 per bin)
RELATIVE_F0_MIN = -1
RELATIVE_F0_MAX = 1
RELATIVE_F0_STEP = 0.01
ONE_OVER_RELATIVE_F0_STEP = int(round(1 / RELATIVE_F0_STEP))
RELATIVE_F0_BINS = 201

# [-120, 0], 1 per bin, "ld" means loudness
ABSOLUTE_LD_MIN = -120
ABSOLUTE_LD_MAX = 0
ABSOLUTE_LD_STEP = 1
ONE_OVER_ABSOLUTE_LD_STEP = int(round(1 / ABSOLUTE_LD_STEP))
ABSOLUTE_LD_BINS = 121

# [-1, 1] in cents (0.01 per bin)
ABSOLUTE_F0_MIN = 24
ABSOLUTE_F0_MAX = 96
ABSOLUTE_F0_STEP = 1 / 5
ONE_OVER_ABSOLUTE_F0_STEP = int(round(1 / ABSOLUTE_F0_STEP))
ABSOLUTE_F0_BINS = 361


def get_onehot_f0_dv(f0_dv):
  f0_dv = tf.clip_by_value(f0_dv[:, :, -1], RELATIVE_F0_MIN,
                           RELATIVE_F0_MAX) - RELATIVE_F0_MIN
  f0_dv = tf.cast(f0_dv / RELATIVE_F0_STEP, tf.int64)
  f0_dv_onehot = tf.one_hot(f0_dv, RELATIVE_F0_BINS)
  return f0_dv_onehot


def get_onehot_ld(ld):
  ld = tf.clip_by_value(ld[:, :, -1], ABSOLUTE_LD_MIN,
                          ABSOLUTE_LD_MAX) - ABSOLUTE_LD_MIN
  ld = tf.cast(ld / ABSOLUTE_LD_STEP, tf.int64)
  ld_onehot = tf.one_hot(ld, ABSOLUTE_LD_BINS)
  return ld_onehot


def get_float_f0_dv(f0_dv_onehot):
  f0_dv = tf.cast(tf.argmax(f0_dv_onehot, axis=-1) /
                  ONE_OVER_RELATIVE_F0_STEP + RELATIVE_F0_MIN,
                  tf.float32)
  return f0_dv[..., tf.newaxis]


def get_float_ld(ld_onehot):
  ld = tf.cast(tf.argmax(ld_onehot, axis=-1) /
                 ONE_OVER_ABSOLUTE_LD_STEP + ABSOLUTE_LD_MIN,
                 tf.float32)
  return ld[..., tf.newaxis]


def get_float_absolute_f0(f0_midi_onehot):
  f0_midi = tf.cast(tf.argmax(f0_midi_onehot, axis=-1) /
                    ONE_OVER_ABSOLUTE_F0_STEP + ABSOLUTE_LD_MIN,
                    tf.float32)
  return f0_midi[..., tf.newaxis]


def get_onehot_absolute_f0(f0_midi):
  f0_midi = tf.clip_by_value(f0_midi[:, :, -1], ABSOLUTE_F0_MIN,
                             ABSOLUTE_F0_MAX) - ABSOLUTE_F0_MIN
  f0_midi = tf.cast(f0_midi / ABSOLUTE_F0_STEP, tf.int64)
  f0_midi_onehot = tf.one_hot(f0_midi, ABSOLUTE_F0_BINS)
  return f0_midi_onehot


def top_p_sample(logits, p=0.95):
  logits_sort = tf.sort(logits, direction='DESCENDING', axis=-1)
  probs_sort = tf.nn.softmax(logits_sort)
  probs_sums = tf.cumsum(probs_sort, axis=1, exclusive=True)
  logits_masked = tf.where(probs_sums < p, logits_sort,
                           tf.ones_like(logits_sort) * 1000)
  min_logits = tf.reduce_min(input_tensor=logits_masked, axis=1, keepdims=True)
  logits = tf.where(
    logits < min_logits,
    tf.ones_like(logits, dtype=logits.dtype) * -1e10,
    logits,
  )
  return logits


def sample_from(logits, return_onehot=False, sample_method='top_p'):
  original_shape = logits.shape
  num_class = logits.shape[-1]
  if sample_method == 'argmax':
    value = tf.argmax(tf.reshape(logits, [-1, num_class]), axis=-1)
  elif sample_method == 'random':
    value = tf.random.categorical(tf.reshape(logits, [-1, num_class]), 1)
  elif sample_method == 'top_p':
    value = tf.random.categorical(
      top_p_sample(tf.reshape(logits, [-1, num_class])), 1)
  if return_onehot:
    value = tf.one_hot(value, num_class)
    return tf.reshape(value, original_shape)
  else:
    return tf.reshape(value, original_shape[:-1] + [1])


class MidiToF0AutoregressiveDecoder(TwoLayerCondAutoregRNN):
  def __init__(self, nhid=256, norm=True, dropout=0.5, sampling_method='top_p'):
    super(MidiToF0AutoregressiveDecoder, self).__init__(
      nhid=nhid,
      n_out=RELATIVE_F0_BINS,
      input_dropout=True,
      input_dropout_p=0.5,
      dropout=dropout,
    )
    self.birnn = tfkl.Bidirectional(tfkl.GRU(
      units=nhid, return_sequences=True, dropout=dropout
    ), )
    self.dense_out = tfkl.Dense(self.n_out)
    self.norm = nn.Normalize('layer') if norm else None
    self.sampling_method = sampling_method

  def split_teacher_force_output(self, output):
    f0_midi_dv_logits = output
    f0_midi_dv_onehot = get_onehot_f0_dv(get_float_f0_dv(f0_midi_dv_logits))
    output = {
      'f0_midi_dv_onehot': f0_midi_dv_onehot,
      'f0_midi_dv_logits': f0_midi_dv_logits,
    }
    return output

  def encode_cond(self, cond, training=False):
    cond = self.birnn(cond, training=training)
    return cond

  def decode_out(self, z_out, training=False):
    if self.norm is not None:
      z_out = self.norm(z_out)
    output = self.dense_out(z_out)
    return output

  def sample_out(self, out, training=False):
    f0_midi_dv_logits = out[..., :RELATIVE_F0_BINS]
    f0_midi_dv_onehot = sample_from(f0_midi_dv_logits,
                                    return_onehot=True,
                                    sample_method=self.sampling_method)
    output = {
      'f0_midi_dv_logits': f0_midi_dv_logits,
      'f0_midi_dv_onehot': f0_midi_dv_onehot,
      'curr_out': f0_midi_dv_onehot,
    }
    return output

  def preprocess(self, q_pitch, out):
    f0_midi = ddsp.core.hz_to_midi(out['f0_hz'])
    f0_midi_dv = f0_midi - q_pitch  # f0 residual
    f0_dv_onehot = get_onehot_f0_dv(f0_midi_dv)
    return f0_dv_onehot

  def postprocess(self, outputs, q_pitch, training=False):
    outputs['f0_midi_dv'] = get_float_f0_dv(outputs['f0_midi_dv_onehot'])
    outputs['f0_midi'] = outputs['f0_midi_dv'] + q_pitch
    outputs['f0_hz'] = ddsp.core.midi_to_hz(outputs['f0_midi'])
    return outputs

  def call(self, q_pitch, z_midi_decoder, conditioning_dict, out=None,
           training=False, display_progressbar=False):
    if training:
      out = self.preprocess(q_pitch, out)
      outputs = self.teacher_force(z_midi_decoder, out, training=training)
    else:
      outputs = self.autoregressive(z_midi_decoder, training=training,
                                    display_progressbar=display_progressbar)

    outputs = self.postprocess(outputs, q_pitch)

    return outputs


class MidiToF0AmpAutoregressiveDecoder(TwoLayerCondAutoregRNN):
  def __init__(self, nhid=256, norm=True, dropout=0.5, sampling_method='top_p'):
    super(MidiToF0AmpAutoregressiveDecoder, self).__init__(
      nhid=nhid,
      n_out=RELATIVE_F0_BINS + ABSOLUTE_LD_BINS,
      input_dropout=True,
      input_dropout_p=0.5,
      dropout=dropout,
    )
    self.birnn = tfkl.Bidirectional(tfkl.GRU(
      units=nhid, return_sequences=True, dropout=dropout
    ), )
    self.dense_out = nn.FcStackOut(ch=nhid, layers=2, n_out=self.n_out)
    self.dense_in = nn.FcStack(ch=nhid, layers=2)
    self.norm = nn.Normalize('layer') if norm else None
    self.sampling_method = sampling_method

  def encode_z(self, z, training=False):
    return self.dense_in(z)

  def split_teacher_force_output(self, output):
    f0_midi_dv_logits = output[..., :RELATIVE_F0_BINS]
    f0_midi_dv_onehot = get_onehot_f0_dv(get_float_f0_dv(f0_midi_dv_logits))
    ld_logits = output[...,
                RELATIVE_F0_BINS:RELATIVE_F0_BINS + ABSOLUTE_LD_BINS]
    ld_onehot = get_onehot_ld(get_float_ld(ld_logits))
    output = {
      'f0_midi_dv_logits': f0_midi_dv_logits,
      'f0_midi_dv_onehot': f0_midi_dv_onehot,
      'ld_logits': ld_logits,
      'ld_onehot': ld_onehot,
    }
    return output

  def encode_cond(self, cond, training=False):
    cond = self.birnn(cond, training=training)
    return cond

  def decode_out(self, z_out, training=False):
    if self.norm is not None:
      z_out = self.norm(z_out)
    output = self.dense_out(z_out)
    return output

  def sample_out(self, out, training=False):
    f0_midi_dv_logits = out[..., :RELATIVE_F0_BINS]
    f0_midi_dv_onehot = sample_from(f0_midi_dv_logits, return_onehot=True,
                                    sample_method=self.sampling_method)
    ld_logits = out[..., RELATIVE_F0_BINS:RELATIVE_F0_BINS + ABSOLUTE_LD_BINS]
    ld_onehot = sample_from(ld_logits, return_onehot=True,
                            sample_method=self.sampling_method)
    output = {
      'f0_midi_dv_logits': f0_midi_dv_logits,
      'f0_midi_dv_onehot': f0_midi_dv_onehot,
      'ld_logits': ld_logits,
      'ld_onehot': ld_onehot,
      'curr_out': tf.concat([f0_midi_dv_onehot, ld_onehot], axis=-1)
    }
    return output

  def preprocess(self, q_pitch, out):
    f0_midi = ddsp.core.hz_to_midi(out['f0_hz'])
    f0_midi_dv = f0_midi - q_pitch  # f0 residual
    f0_midi_dv_onehot = get_onehot_f0_dv(f0_midi_dv)
    ld = out['loudness_db']
    ld_onehot = get_onehot_ld(ld)
    return tf.concat([f0_midi_dv_onehot, ld_onehot], axis=-1)

  def postprocess(self, outputs, q_pitch, training=False):
    outputs['f0_midi_dv'] = get_float_f0_dv(outputs['f0_midi_dv_onehot'])
    outputs['f0_midi'] = outputs['f0_midi_dv'] + q_pitch
    outputs['f0_hz'] = ddsp.core.midi_to_hz(outputs['f0_midi'])
    outputs['ld'] = get_float_ld(outputs['ld_onehot'])
    return outputs

  def call(self, q_pitch, z_midi_decoder, out=None, training=False):

    if training:
      out = self.preprocess(q_pitch, out)
      outputs = self.teacher_force(z_midi_decoder, out, training=training)
    else:
      outputs = self.autoregressive(z_midi_decoder, training=training)

    outputs = self.postprocess(outputs, q_pitch)

    return outputs

  def partial_teacher_force(self, q_pitch, z_midi_decoder, out,
                            teacher_force_mask, training=False,
                            display_progressbar=False):
    # right shift and encode out
    out = self.preprocess(q_pitch, out)
    go_frame = tf.tile([[0.0]], [out.shape[0], out.shape[-1]])[:, tf.newaxis, :]
    out = tf.concat([go_frame, out[:, :-1, :]], axis=1)
    out = self.encode_out(out)

    cond = self.encode_cond(z_midi_decoder, training=training)
    batch_size = cond.shape[0]
    length = cond.shape[1]
    prev_out = tf.tile([[0.0]], [batch_size, self.n_out])[:, tf.newaxis,
               :]  # go_frame
    prev_states = (None, None)
    overall_outputs = []

    teacher_force_batch_size = teacher_force_mask.shape[0]

    for i in tqdm(range(length), position=0, leave=True, desc="Generating: ",
                  disable=not display_progressbar):
      curr_cond = tf.tile(cond[:, i:i + 1, :], [teacher_force_batch_size, 1, 1])
      prev_out = self.encode_out(prev_out) * (
            1 - teacher_force_mask[:, i:i + 1, :]) + \
                 out[:, i:i + 1, :] * teacher_force_mask[:, i:i + 1, :]
      curr_out, curr_states = self._one_step(curr_cond, prev_out, prev_states,
                                             training=training)
      curr_out = self.sample_out(curr_out)
      overall_outputs.append(curr_out)
      prev_out, prev_states = curr_out['curr_out'], curr_states

    outputs = {}
    for k in curr_out.keys():
      outputs[k] = tf.concat([x[k] for x in overall_outputs], axis=1)

    return self.postprocess(outputs, q_pitch)


class F0AmpAutoregressiveDecoder(TwoLayerCondAutoregRNN):  # unconditional
  def __init__(self, nhid=256, norm=True, dropout=0.5, sampling_method='top_p'):
    self.n_f0 = ABSOLUTE_F0_BINS
    self.n_ld = ABSOLUTE_LD_BINS
    super(F0AmpAutoregressiveDecoder, self).__init__(
      nhid=nhid,
      n_out=self.n_f0 + self.n_ld,
      input_dropout=True,
      input_dropout_p=0.5,
      dropout=dropout,
    )
    self.birnn = tfkl.Bidirectional(tfkl.GRU(
      units=nhid, return_sequences=True, dropout=dropout
    ), )
    self.dense_out = nn.FcStackOut(ch=nhid, layers=2, n_out=self.n_out)
    self.dense_in = nn.FcStack(ch=nhid, layers=2)
    self.norm = nn.Normalize('layer') if norm else None
    self.sampling_method = sampling_method

  def split_teacher_force_output(self, output):
    # split outputs
    f0_midi_logits = output[..., :self.n_f0]
    f0_midi_onehot = get_onehot_absolute_f0(
      get_float_absolute_f0(f0_midi_logits))
    ld_logits = output[..., self.n_f0:self.n_f0 + self.n_ld]
    ld_onehot = get_onehot_ld(get_float_ld(ld_logits))
    output = {
      'f0_midi_logits': f0_midi_logits,
      'f0_midi_onehot': f0_midi_onehot,
      'ld_logits': ld_logits,
      'ld_onehot': ld_onehot,
    }
    return output

  def encode_z(self, z, training=False):
    return self.dense_in(z)

  def decode_out(self, z_out, training=False):
    if self.norm is not None:
      z_out = self.norm(z_out)
    output = self.dense_out(z_out)
    return output

  def sample_out(self, out, training=False):
    f0_midi_logits = out[..., :self.n_f0]
    f0_midi_onehot = sample_from(f0_midi_logits, return_onehot=True,
                                 sample_method=self.sampling_method)
    ld_logits = out[..., self.n_f0:self.n_f0 + self.n_ld]
    ld_onehot = sample_from(ld_logits, return_onehot=True,
                            sample_method=self.sampling_method)
    output = {
      'f0_midi_logits': f0_midi_logits,
      'f0_midi_onehot': f0_midi_onehot,
      'ld_logits': ld_logits,
      'ld_onehot': ld_onehot,
      'curr_out': tf.concat([f0_midi_onehot, ld_onehot], axis=-1)
    }
    return output

  def preprocess(self, out, training=False):
    f0_midi = ddsp.core.hz_to_midi(out['f0_hz'])
    f0_midi_onehot = get_onehot_absolute_f0(f0_midi)
    ld = out['loudness_db']
    ld_onehot = get_onehot_ld(ld)
    return tf.concat([f0_midi_onehot, ld_onehot], axis=-1)

  def postprocess(self, outputs, training=False):
    outputs['f0_midi'] = get_float_absolute_f0(outputs['f0_midi_onehot'])
    outputs['f0_hz'] = ddsp.core.midi_to_hz(outputs['f0_midi'])
    outputs['ld'] = get_float_ld(outputs['ld_onehot'])
    return outputs

  def call(self, q_pitch, z_midi_decoder, out=None, training=False):
    z_midi_decoder = z_midi_decoder[..., 66:]

    if training:
      out = self.preprocess(out)
      outputs = self.teacher_force(z_midi_decoder, out, training=training)
    else:
      outputs = self.autoregressive(z_midi_decoder, training=training)

    outputs = self.postprocess(outputs)

    return outputs


class MidiF0ToHarmonicDecoder(nn.DictLayer):  # TODO: (yusongwu) merge into DDSP
  def __init__(self,
               net=None,
               norm=True,
               output_splits=(('f0_midi', 1),
                              ('amplitudes', 1),
                              ('harmonic_distribution', 60),
                              ('magnitudes', 65)),
               **kwargs):
    """Constructor."""
    self.output_splits = output_splits
    self.n_out = sum([v[1] for v in output_splits])
    output_keys = [v[0] for v in output_splits]
    super().__init__(output_keys=output_keys, **kwargs)

    # Layers.
    self.net = net
    self.dense_out = tfkl.Dense(self.n_out)
    self.norm = nn.Normalize('layer') if norm else None

  def call(self, z, training=False):
    x = self.net(z, training=training)

    if self.norm is not None:
      x = self.norm(x)

    x = self.dense_out(x)

    outputs = nn.split_to_dict(x, self.output_splits)
    return outputs


class MidiToHarmonicAutoregressiveDecoder(tfkl.Layer):
  def __init__(self, nhid=256,
               hd_noise_output_splits=(('amplitudes', 1),
                                       ('harmonic_distribution', 60),
                                       ('noise_magnitudes', 65)),
               net_type='conv'):
    super(MidiToHarmonicAutoregressiveDecoder, self).__init__()
    self.q_pitch_emb = tfkl.Dense(64)
    self.midi_to_f0 = MidiToF0AutoregressiveDecoder(nhid=nhid)
    self.net_type = net_type
    if self.net_type == 'conv':
      self.midi_f0_to_harmonic = MidiF0ToHarmonicDecoder(
        net=nn.DilatedConvStack(
          ch=128,
          layers_per_stack=5,
          stacks=4,
          norm_type='layer',
          conditional=False,
        ),
        output_splits=hd_noise_output_splits,
      )
    elif self.net_type == 'noise_conv':
      self.midi_f0_to_harmonic = MidiF0ToHarmonicDecoder(
        net=nn.DilatedConvStack(
          ch=128,
          layers_per_stack=5,
          stacks=4,
          norm_type='layer',
          conditional=True,
        ),
        output_splits=hd_noise_output_splits,
      )

  def call(self, q_pitch, z_midi_decoder, conditioning_dict, out=None,
           training=False, display_progressbar=False):
    # z_midi_decoder = self.cond_birnn(z_midi_decoder)
    f0_output = self.midi_to_f0(q_pitch, z_midi_decoder, conditioning_dict,
                                out=out, training=training,
                                display_progressbar=display_progressbar)

    if training:
      f0_midi = ddsp.core.hz_to_midi(out['f0_hz'])
    else:
      f0_midi = f0_output['f0_midi']

    if self.net_type == 'conv':
      z = tf.concat([z_midi_decoder, self.q_pitch_emb(f0_midi / 127)], axis=-1)
      outputs = self.midi_f0_to_harmonic(z, training=training)
      hd_noise_output = outputs
    elif self.net_type == 'noise_conv':
      z = tf.concat([z_midi_decoder, self.q_pitch_emb(f0_midi / 127)], axis=-1)
      noise = tf.random.normal([z.shape[0], z.shape[1], 100])
      outputs = self.midi_f0_to_harmonic([noise, z], training=training)
      hd_noise_output = outputs

    output = hd_noise_output
    output['prenet_f0_output'] = f0_output
    if training:
      output['f0_hz'] = out['f0_hz']
    else:
      output['f0_hz'] = ddsp.core.midi_to_hz(f0_output['f0_midi'],
                                             midi_zero_silence=True)

    return output


class MidiNoiseToHarmonicDecoder(nn.DictLayer):
  """Decodes MIDI notes (& velocities) to f0, amps, hd, noise."""

  def __init__(self,
               net=None,
               f0_residual=True,
               norm=True,
               output_splits=(('f0_midi', 1),
                              ('amplitudes', 1),
                              ('harmonic_distribution', 60),
                              ('magnitudes', 65)),
               **kwargs):
    """Constructor."""
    self.output_splits = output_splits
    self.n_out = sum([v[1] for v in output_splits])
    output_keys = [v[0] for v in output_splits] + ['f0_hz']
    super().__init__(output_keys=output_keys, **kwargs)

    # Layers.
    self.net = net
    self.f0_residual = f0_residual
    self.dense_out = tfkl.Dense(self.n_out)
    self.norm = nn.Normalize('layer') if norm else None

  def call(self, noise, z_pitch, z=None):
    """Forward pass for the MIDI decoder.

    Args:
      z_pitch: Tensor containing encoded pitch in MIDI scale. [batch, time, 1].
      z_vel: Tensor containing encoded velocity in MIDI scale. [batch, time, 1].
      z: Additional non-MIDI latent tensor. [batch, time, n_z]

    Returns:
      A dictionary to feed into a processor group.
    """
    # pylint: disable=unused-argument
    # x = tf.concat([z_pitch, z_vel], axis=-1)  # TODO(jesse): Allow velocity.
    x = noise
    x = self.net(x) if z is None else self.net([x, z])

    if self.norm is not None:
      x = self.norm(x)

    x = self.dense_out(x)

    outputs = nn.split_to_dict(x, self.output_splits)

    if self.f0_residual:
      outputs['f0_midi'] += z_pitch

    outputs['f0_hz'] = ddsp.core.midi_to_hz(outputs['f0_midi'])
    return outputs
