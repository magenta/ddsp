# Copyright 2019 The DDSP Authors.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ddsp
from ddsp.training import nn
import gin
import numpy as np
import tensorflow.compat.v1 as tf

tfkl = tf.keras.layers


# ------------------ Encoders --------------------------------------------------
class Encoder(object):
  """Base class to implement any encoder.

  Users should override compute_z() to define the actual encoder structure.
  Optionally, set infer_f0 to True and override compute_f0.
  Hyper-parameters will be passed through the constructor.
  """

  def __init__(self, name, f0_encoder=None):
    self.name = name
    self.f0_encoder = f0_encoder

  def __call__(self, conditioning):
    return self.get_outputs(conditioning)

  def get_outputs(self, conditioning):
    """Updates conditioning with z and (optionally) f0."""
    if self.f0_encoder:
      # conditioning['f0'] is a value in [0, 1]
      # corresponding to midi values [0..127]
      conditioning['f0'] = self.f0_encoder(conditioning)
      conditioning['f0_hz'] = ddsp.midi_to_hz(conditioning['f0'] * 127.0)

    z = self.compute_z(conditioning)
    time_steps = int(conditioning['f0'].shape[1])
    conditioning['z'] = self.expand_z(z, time_steps)

    return conditioning

  def expand_z(self, z, time_steps):
    """Make sure z has same temporal resolution as other conditioning."""
    # Add time dim of z if necessary.
    if len(z.shape) == 2:
      z = z[:, tf.newaxis, :]
    # Expand time dim of z if necessary.
    z_time_steps = int(z.shape[1])
    if z_time_steps != time_steps:
      z = ddsp.resample(z, time_steps)
    return z

  def compute_z(self, conditioning):
    raise NotImplementedError


@gin.configurable
class MfccTimeDistributedRnnEncoder(Encoder):
  """Use MFCCs as latent variables, distribute across timesteps."""

  def __init__(self,
               rnn_channels=512,
               rnn_type='gru',
               z_dims=32,
               z_time_steps=250,
               name='mfcc_rnn_encoder',
               f0_encoder=None):
    super(MfccTimeDistributedRnnEncoder, self).__init__(name=name,
                                                        f0_encoder=f0_encoder)
    self.rnn_channels = rnn_channels
    self.rnn_type = rnn_type
    self.z_dims = z_dims
    if z_time_steps not in [63, 125, 250, 500, 1000]:
      raise ValueError(
          '`z_time_steps` currently limited to 63,125,250,500 and 1000')
    else:
      self.z_time_steps = z_time_steps
    self.z_audio_spec = {
        63: {
            'fft_size': 2048,
            'overlap': 0.5
        },
        125: {
            'fft_size': 1024,
            'overlap': 0.5
        },
        250: {
            'fft_size': 1024,
            'overlap': 0.75
        },
        500: {
            'fft_size': 512,
            'overlap': 0.75
        },
        1000: {
            'fft_size': 256,
            'overlap': 0.75
        }
    }
    self.fft_size = self.z_audio_spec[self.z_time_steps]['fft_size']
    self.overlap = self.z_audio_spec[self.z_time_steps]['overlap']

  def compute_z(self, conditioning):
    mfccs = ddsp.spectral_ops.calc_mfcc(
        conditioning['audio'],
        lo_hz=20.0,
        hi_hz=8000.0,
        fft_size=self.fft_size,
        mel_bins=128,
        mfcc_bins=30,
        overlap=self.overlap,
        pad_end=True)

    z = nn.Normalize('instance')(mfccs[:, :, tf.newaxis, :])[:, :, 0, :]

    rnn = {'lstm': tfkl.LSTM, 'gru': tfkl.GRU}[self.rnn_type]
    # Retain hidden outputs at each time step.
    z = rnn(self.rnn_channels, return_sequences=True)(z)
    # Bounce down to compressed z dimensions.
    z = tfkl.Dense(self.z_dims)(z)
    return z


class F0EncoderMixin(object):
  """Mixin for F0 encoders."""

  def __call__(self, conditioning):
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      return self.compute_f0(conditioning)

  def _compute_unit_midi(self, probs):
    """Computes the midi from a distribution over the unit interval."""
    # probs: [B, T, D]
    depth = int(probs.shape[-1])

    unit_midi_bins = tf.constant(
        1.0 * np.arange(depth).reshape((1, 1, -1)) / depth,
        dtype=tf.float32)  # [1, 1, D]

    f0_unit_midi = tf.reduce_sum(
        unit_midi_bins * probs, axis=-1, keepdims=True)  # [B, T, 1]
    return f0_unit_midi


@gin.configurable
class ResnetF0Encoder(F0EncoderMixin):
  """Embeddings from resnet on spectrograms."""

  def __init__(self,
               size='large',
               f0_bins=128,
               spectral_fn=lambda x: ddsp.spectral_ops.calc_mag(x, size=1024),
               name='resnet_f0_encoder'):
    self.size = size
    self.f0_bins = f0_bins
    self.spectral_fn = spectral_fn
    self.name = name

  def compute_f0(self, conditioning):
    """Compute fundamental frequency."""
    mag = self.spectral_fn(conditioning['audio'])
    mag = mag[:, :, :, tf.newaxis]
    x = nn.resnet(mag, size=self.size)

    # Collapse the frequency dimension
    x_shape = x.shape.as_list()
    y = tf.reshape(x, [x_shape[0], x_shape[1], -1])
    # Project to f0_bins
    y = tfkl.Dense(self.f0_bins)(y)

    # treat the NN output as probability over midi values.
    # probs = tf.nn.softmax(y)  # softmax leads to NaNs
    probs = tf.nn.softplus(y) + 1e-3
    probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)
    f0 = self._compute_unit_midi(probs)

    # Make same time resolution as original CREPE f0.
    n_timesteps = int(conditioning['f0'].shape[1])
    f0 = ddsp.resample(f0, n_timesteps)
    return f0
