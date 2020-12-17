# Copyright 2020 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Library of encoder objects."""

import ddsp
from ddsp import spectral_ops
from ddsp.training import nn
import gin
import tensorflow.compat.v2 as tf

tfkl = tf.keras.layers


# ------------------ Encoders --------------------------------------------------
class ZEncoder(nn.DictLayer):
  """Base class to implement an encoder that creates a latent z vector.

  Users should override compute_z() to define the actual encoder structure.
  Input_keys from compute_z() instead of call(), output_keys are always ['z'].
  """

  def __init__(self, input_keys=None, **kwargs):
    """Constructor."""
    input_keys = input_keys or self.get_argument_names('compute_z')
    super().__init__(input_keys, output_keys=['z'], **kwargs)

    # TODO(jesseengel): remove dependence on arbitrary key.
    self.input_keys.append('f0_scaled')  # Input to get n_timesteps dynamically.

  def call(self, *args, **unused_kwargs):
    """Takes in input tensors and returns a latent tensor z."""
    time_steps = int(args[-1].shape[1])
    inputs = args[:-1]  # Last input just used for time_steps.
    z = self.compute_z(*inputs)
    return self.expand_z(z, time_steps)

  def expand_z(self, z, time_steps):
    """Make sure z has same temporal resolution as other conditioning."""
    # Add time dim of z if necessary.
    if len(z.shape) == 2:
      z = z[:, tf.newaxis, :]
    # Expand time dim of z if necessary.
    z_time_steps = int(z.shape[1])
    if z_time_steps != time_steps:
      z = ddsp.core.resample(z, time_steps)
    return z

  def compute_z(self, *inputs):
    """Takes in input tensors and returns a latent tensor z."""
    raise NotImplementedError


@gin.register
class MfccTimeDistributedRnnEncoder(ZEncoder):
  """Use MFCCs as latent variables, distribute across timesteps."""

  def __init__(self,
               rnn_channels=512,
               rnn_type='gru',
               z_dims=32,
               z_time_steps=250,
               **kwargs):
    super().__init__(**kwargs)
    if z_time_steps not in [63, 125, 250, 500, 1000]:
      raise ValueError(
          '`z_time_steps` currently limited to 63,125,250,500 and 1000')
    self.z_audio_spec = {
        '63': {
            'fft_size': 2048,
            'overlap': 0.5
        },
        '125': {
            'fft_size': 1024,
            'overlap': 0.5
        },
        '250': {
            'fft_size': 1024,
            'overlap': 0.75
        },
        '500': {
            'fft_size': 512,
            'overlap': 0.75
        },
        '1000': {
            'fft_size': 256,
            'overlap': 0.75
        }
    }
    self.fft_size = self.z_audio_spec[str(z_time_steps)]['fft_size']
    self.overlap = self.z_audio_spec[str(z_time_steps)]['overlap']

    # Layers.
    self.z_norm = nn.Normalize('instance')
    self.rnn = nn.Rnn(rnn_channels, rnn_type)
    self.dense_out = tfkl.Dense(z_dims)

  def compute_z(self, audio):
    mfccs = spectral_ops.compute_mfcc(
        audio,
        lo_hz=20.0,
        hi_hz=8000.0,
        fft_size=self.fft_size,
        mel_bins=128,
        mfcc_bins=30,
        overlap=self.overlap,
        pad_end=True)

    # Normalize.
    z = self.z_norm(mfccs[:, :, tf.newaxis, :])[:, :, 0, :]
    # Run an RNN over the latents.
    z = self.rnn(z)
    # Bounce down to compressed z dimensions.
    z = self.dense_out(z)
    return z


# Transcribing Autoencoder Encoders --------------------------------------------
@gin.register
class ResnetSinusoidalEncoder(nn.DictLayer):
  """This encoder maps directly from audio to synthesizer parameters.

  EXPERIMENTAL

  It is equivalent of a base Encoder and Decoder together.
  """

  def __init__(self,
               output_splits=(('frequencies', 100 * 64),
                              ('amplitudes', 100),
                              ('noise_magnitudes', 60)),
               spectral_fn=spectral_ops.compute_logmel,
               size='tiny',
               **kwargs):
    super().__init__(output_keys=[key for key, dim in output_splits], **kwargs)
    self.output_splits = output_splits
    self.spectral_fn = spectral_fn

    # Layers.
    self.resnet = nn.ResNet(size=size)
    self.dense_outs = [tfkl.Dense(v[1]) for v in output_splits]

  def call(self, audio):
    """Updates conditioning with z and (optionally) f0."""
    outputs = {}

    # [batch, 64000, 1]
    mag = self.spectral_fn(audio)

    # [batch, 125, 229]
    mag = mag[:, :, :, tf.newaxis]
    x = self.resnet(mag)

    # [batch, 125, 8, 1024]
    # # Collapse the frequency dimension.
    x = tf.reshape(x, [int(x.shape[0]), int(x.shape[1]), -1])

    # [batch, 125, 8192]
    for layer, key in zip(self.dense_outs, self.output_keys):
      outputs[key] = layer(x)

    return outputs


@gin.register
class SinusoidalToHarmonicEncoder(nn.DictLayer):
  """Predicts harmonic controls from sinusoidal controls.

  EXPERIMENTAL
  """

  def __init__(self,
               net=None,
               n_harmonics=100,
               f0_depth=64,
               amp_scale_fn=ddsp.core.exp_sigmoid,
               # pylint: disable=g-long-lambda
               freq_scale_fn=lambda x: ddsp.core.frequencies_softmax(
                   x, depth=64, hz_min=20.0, hz_max=1200.0),
               # pylint: enable=g-long-lambda
               sample_rate=16000,
               **kwargs):
    """Constructor."""
    super().__init__(**kwargs)
    self.n_harmonics = n_harmonics
    self.amp_scale_fn = amp_scale_fn
    self.freq_scale_fn = freq_scale_fn
    self.sample_rate = sample_rate

    # Layers.
    self.net = net
    self.amp_out = tfkl.Dense(1)
    self.hd_out = tfkl.Dense(n_harmonics)
    self.f0_out = tfkl.Dense(f0_depth)

  def call(self, sin_freqs, sin_amps) -> ['harm_amp', 'harm_dist', 'f0_hz']:
    """Converts (sin_freqs, sin_amps) to (f0, amp, hd).

    Args:
      sin_freqs: Sinusoidal frequencies in Hertz, of shape
        [batch, time, n_sinusoids].
      sin_amps: Sinusoidal amplitudes, linear scale, greater than 0, of shape
        [batch, time, n_sinusoids].

    Returns:
      f0: Fundamental frequency in Hertz, of shape [batch, time, 1].
      amp: Amplitude, linear scale, greater than 0, of shape [batch, time, 1].
      hd: Harmonic distribution, linear scale, greater than 0, of shape
        [batch, time, n_harmonics].
    """
    # Scale the inputs.
    nyquist = self.sample_rate / 2.0
    sin_freqs_unit = ddsp.core.hz_to_unit(sin_freqs, hz_min=0.0, hz_max=nyquist)

    # Combine.
    x = tf.concat([sin_freqs_unit, sin_amps], axis=-1)

    # Run it through the network.
    x = self.net(x)
    x = x['out'] if isinstance(x, dict) else x

    # Output layers.
    harm_amp = self.amp_out(x)
    harm_dist = self.hd_out(x)
    f0 = self.f0_out(x)

    # Output scaling.
    harm_amp = self.amp_scale_fn(harm_amp)
    harm_dist = self.amp_scale_fn(harm_dist)
    f0_hz = self.freq_scale_fn(f0)

    # Filter harmonic distribution for nyquist.
    harm_freqs = ddsp.core.get_harmonic_frequencies(f0_hz, self.n_harmonics)
    harm_dist = ddsp.core.remove_above_nyquist(harm_freqs,
                                               harm_dist,
                                               self.sample_rate)
    harm_dist = ddsp.core.safe_divide(
        harm_dist, tf.reduce_sum(harm_dist, axis=-1, keepdims=True))

    return (harm_amp, harm_dist, f0_hz)


