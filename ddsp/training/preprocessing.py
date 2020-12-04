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
"""Library of preprocess functions."""

import ddsp
from ddsp.training import nn
import gin
import tensorflow as tf

hz_to_midi = ddsp.core.hz_to_midi
F0_RANGE = ddsp.spectral_ops.F0_RANGE
LD_RANGE = ddsp.spectral_ops.LD_RANGE

tfkl = tf.keras.layers


# ---------------------- Preprocess Helpers ------------------------------------
def at_least_3d(x):
  """Optionally adds time, batch, then channel dimension."""
  x = x[tf.newaxis] if not x.shape else x
  x = x[tf.newaxis, :] if len(x.shape) == 1 else x
  x = x[:, :, tf.newaxis] if len(x.shape) == 2 else x
  return x


# ---------------------- Preprocess objects ------------------------------------
@gin.register
class F0LoudnessPreprocessor(nn.DictLayer):
  """Resamples and scales 'f0_hz' and 'loudness_db' features."""

  def __init__(self, time_steps=1000, **kwargs):
    super().__init__(**kwargs)
    self.time_steps = time_steps

  def call(self, loudness_db, f0_hz, **unused_kwargs) -> [
      'f0_hz', 'loudness_db', 'f0_scaled', 'ld_scaled']:
    # Resample features to the frame_rate.
    f0_hz = at_least_3d(f0_hz)
    loudness_db = at_least_3d(loudness_db)
    f0_resampled = ddsp.core.resample(f0_hz, self.time_steps)
    ld_resampled = ddsp.core.resample(loudness_db, self.time_steps)
    # For NN training, scale frequency and loudness to the range [0, 1].
    # Log-scale f0 features. Loudness from [-1, 0] to [1, 0].
    f0_scaled = hz_to_midi(f0_resampled) / F0_RANGE
    ld_scaled = (ld_resampled / LD_RANGE) + 1.0
    return f0_hz, loudness_db, f0_scaled, ld_scaled

  @staticmethod
  def invert_scaling(f0_scaled, ld_scaled):
    """Takes in scaled f0 and loudness, and puts them back to hz & db scales."""
    f0_hz = ddsp.core.midi_to_hz(F0_RANGE * f0_scaled)
    loudness_db = (ld_scaled - 1.0) * LD_RANGE
    return f0_hz, loudness_db


@gin.register
class F0PowerPreprocessor(F0LoudnessPreprocessor):
  """Dynamically compute additional power_db feature."""

  def __init__(self,
               time_steps=1000,
               frame_rate=250,
               sample_rate=16000,
               frame_size=64,
               **kwargs):
    super().__init__(time_steps, **kwargs)
    self.frame_rate = frame_rate
    self.sample_rate = sample_rate
    self.frame_size = frame_size

  def call(self, audio, loudness_db, f0_hz, **kwargs) -> [
      'f0_hz', 'loudness_db', 'f0_scaled', 'ld_scaled', 'pw_scaled', 'pw_db']:
    """Compute power on the fly."""
    pw_db = ddsp.spectral_ops.compute_power(audio,
                                            sample_rate=self.sample_rate,
                                            frame_rate=self.frame_rate,
                                            frame_size=self.frame_size)
    if pw_db.shape[1] != self.time_steps:
      raise ValueError(f'Preprocessor: Power time_steps {pw_db.shape[1]} '
                       f'is not the same as self.time_steps {self.time_steps}.')

    pw_scaled = (pw_db / LD_RANGE) + 1.0
    f0_hz, loudness_db, f0_scaled, ld_scaled = super().call(
        loudness_db, f0_hz, **kwargs)
    return f0_hz, loudness_db, f0_scaled, ld_scaled, pw_scaled, pw_db

  @staticmethod
  def invert_scaling(f0_scaled, ld_scaled, pw_scaled):
    """Puts scaled f0, loudness, and power back to hz & db scales."""
    f0_hz = ddsp.core.midi_to_hz(F0_RANGE * f0_scaled)
    loudness_db = (ld_scaled - 1.0) * LD_RANGE
    power_db = (pw_scaled - 1.0) * LD_RANGE
    return f0_hz, loudness_db, power_db
