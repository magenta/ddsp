# Copyright 2024 The DDSP Authors.
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

"""Library of preprocess functions."""

import ddsp
from ddsp.training import nn
import gin
import tensorflow as tf

F0_RANGE = ddsp.spectral_ops.F0_RANGE
DB_RANGE = ddsp.spectral_ops.DB_RANGE

tfkl = tf.keras.layers


# ---------------------- Preprocess Helpers ------------------------------------
def at_least_3d(x):
  """Optionally adds time, batch, then channel dimension."""
  x = x[tf.newaxis] if not x.shape else x
  x = x[tf.newaxis, :] if len(x.shape) == 1 else x
  x = x[:, :, tf.newaxis] if len(x.shape) == 2 else x
  return x


def scale_db(db):
  """Scales [-DB_RANGE, 0] to [0, 1]."""
  return (db / DB_RANGE) + 1.0


def inv_scale_db(db_scaled):
  """Scales [0, 1] to [-DB_RANGE, 0]."""
  return (db_scaled - 1.0) * DB_RANGE


def scale_f0_hz(f0_hz):
  """Scales [0, Nyquist] Hz to [0, 1.0] MIDI-scaled."""
  return ddsp.core.hz_to_midi(f0_hz) / F0_RANGE


def inv_scale_f0_hz(f0_scaled):
  """Scales [0, 1.0] MIDI-scaled to [0, Nyquist] Hz."""
  return ddsp.core.midi_to_hz(f0_scaled * F0_RANGE)


# ---------------------- Preprocess objects ------------------------------------
@gin.register
class F0LoudnessPreprocessor(nn.DictLayer):
  """Resamples and scales 'f0_hz' and 'loudness_db' features."""

  def __init__(self,
               time_steps=1000,
               frame_rate=250,
               sample_rate=16000,
               compute_loudness=True,
               **kwargs):
    super().__init__(**kwargs)
    self.time_steps = time_steps
    self.frame_rate = frame_rate
    self.sample_rate = sample_rate
    self.compute_loudness = compute_loudness

  def call(self, loudness_db, f0_hz, audio=None) -> [
      'f0_hz', 'loudness_db', 'f0_scaled', 'ld_scaled']:
    # Compute loudness fresh (it's fast).
    if self.compute_loudness:
      loudness_db = ddsp.spectral_ops.compute_loudness(
          audio,
          sample_rate=self.sample_rate,
          frame_rate=self.frame_rate)

    # Resample features to the frame_rate.
    f0_hz = self.resample(f0_hz)
    loudness_db = self.resample(loudness_db)
    # For NN training, scale frequency and loudness to the range [0, 1].
    # Log-scale f0 features. Loudness from [-1, 0] to [1, 0].
    f0_scaled = scale_f0_hz(f0_hz)
    ld_scaled = scale_db(loudness_db)
    return f0_hz, loudness_db, f0_scaled, ld_scaled

  @staticmethod
  def invert_scaling(f0_scaled, ld_scaled):
    """Takes in scaled f0 and loudness, and puts them back to hz & db scales."""
    f0_hz = inv_scale_f0_hz(f0_scaled)
    loudness_db = inv_scale_db(ld_scaled)
    return f0_hz, loudness_db

  def resample(self, x):
    x = at_least_3d(x)
    return ddsp.core.resample(x, self.time_steps)


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

  def call(self, f0_hz, power_db=None, audio=None) -> [
      'f0_hz', 'pw_db', 'f0_scaled', 'pw_scaled']:
    """Compute power on the fly if it's not in the inputs."""
    # For NN training, scale frequency and loudness to the range [0, 1].
    f0_hz = self.resample(f0_hz)
    f0_scaled = scale_f0_hz(f0_hz)

    if power_db is not None:
      # Use dataset values if present.
      pw_db = power_db
    elif audio is not None:
      # Otherwise, compute power on the fly.
      pw_db = ddsp.spectral_ops.compute_power(audio,
                                              sample_rate=self.sample_rate,
                                              frame_rate=self.frame_rate,
                                              frame_size=self.frame_size)
    else:
      raise ValueError('Power preprocessing requires either '
                       '"power_db" or "audio" keys to be provided '
                       'in the dataset.')
    # Resample.
    pw_db = self.resample(pw_db)
    # Scale power.
    pw_scaled = scale_db(pw_db)

    return f0_hz, pw_db, f0_scaled, pw_scaled

  @staticmethod
  def invert_scaling(f0_scaled, pw_scaled):
    """Puts scaled f0, loudness, and power back to hz & db scales."""
    f0_hz = inv_scale_f0_hz(f0_scaled)
    power_db = inv_scale_db(pw_scaled)
    return f0_hz, power_db


@gin.register
class OnlineF0PowerPreprocessor(nn.DictLayer):
  """Dynamically compute power_db and f0_hz with optional centered frames."""

  def __init__(self,
               frame_rate=250,
               frame_size=1024,
               padding='center',
               compute_power=True,
               compute_f0=True,
               crepe_saved_model_path='full',
               viterbi=False,
               **kwargs):
    super().__init__(**kwargs)
    # Preprocessing must happen at 16kHz because CREPE trained at 16kHz.
    self.sample_rate = ddsp.spectral_ops.CREPE_SAMPLE_RATE
    self.frame_rate = frame_rate
    self.frame_size = frame_size
    self.hop_size = self.sample_rate // frame_rate

    self.compute_f0 = compute_f0
    self.compute_power = compute_power

    self.padding = padding

    # Crepe model, must either be a model size or path to SavedModel.
    if crepe_saved_model_path:
      self.crepe_model = ddsp.spectral_ops.PretrainedCREPE(
          model_size_or_path=crepe_saved_model_path, hop_size=self.hop_size
      )

    # Use viterbi decoding.
    self.viterbi = viterbi

  def call(
      self, audio, f0_hz=None, f0_confidence=None, audio_16k=None, pw_db=None
  ) -> ['f0_hz', 'pw_db', 'f0_scaled', 'pw_scaled', 'f0_confidence']:
    """Compute power on the fly if it's not in the inputs."""
    # Compute features at 16kHz (needed for CREPE).
    if audio_16k is not None:
      audio = audio_16k

    # Compute power and f0 on the fly.
    if self.compute_power:
      pw_db = ddsp.spectral_ops.compute_power(audio,
                                              sample_rate=self.sample_rate,
                                              frame_rate=self.frame_rate,
                                              frame_size=self.frame_size,
                                              padding=self.padding)

    if self.compute_f0:
      f0_hz, f0_confidence = self.crepe_model.predict_f0_and_confidence(
          audio, viterbi=self.viterbi, padding=self.padding)
      # Stop gradients from flowing to CREPE.
      f0_hz = tf.stop_gradient(f0_hz)
      f0_confidence = tf.stop_gradient(f0_confidence)
    elif f0_hz is None or f0_confidence is None:
      raise ValueError('Preprocessor must either have `compute_f0=True`, or'
                       '__call__ must be supplied 3 arguments, '
                       '[audio, f0_hz, and f0_confidence].')

    # For NN training, scale frequency and loudness to the range [0, 1].
    pw_db = at_least_3d(pw_db)
    f0_hz = at_least_3d(f0_hz)

    pw_scaled = scale_db(pw_db)
    f0_scaled = scale_f0_hz(f0_hz)

    # For sanity checking.
    # You need to define time_steps correctly just to make sure you know what
    # you're doing, and what the model output shape is (no interpolation).
    n_t = audio.shape[1]
    time_steps, _ = ddsp.spectral_ops.get_framed_lengths(
        n_t, self.frame_size, self.hop_size, self.padding)
    for k, output in {
        'f0_hz': f0_hz,
        'pw_db': pw_db,
        'f0_scaled': f0_scaled,
        'pw_scaled': pw_scaled,
        'f0_confidence': f0_confidence}.items():
      if output.shape[1] != time_steps:
        raise ValueError(
            f'OnlineF0PowerPreprocessor output: ({k}) does not have '
            f'{time_steps} timesteps. Output shape: {output.shape}. '
            f'\nInputs: seconds ({n_t/self.sample_rate}), '
            f'frame_rate ({self.frame_rate}), '
            f'padding ("{self.padding}").')

    return f0_hz, pw_db, f0_scaled, pw_scaled, f0_confidence


