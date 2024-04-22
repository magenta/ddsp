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

"""Functions to generate self-supervised signal, EXPERIMENTAL."""
import warnings

import ddsp
import gin
import numpy as np
import tensorflow.compat.v2 as tf

warnings.warn('Imported synthetic_data.py module, which is EXPERIMENTAL '
              'and likely to change.')


def flip(p=0.5):
  return np.random.uniform() <= p


def uniform_int(minval=0, maxval=10):
  return np.random.random_integers(int(minval), int(maxval))


def uniform_float(minval=0.0, maxval=10.0):
  return np.random.uniform(float(minval), float(maxval))


def uniform_generator(sample_shape, n_timesteps, minval, maxval,
                      method='linear'):
  """Linearly interpolates between a fixed number of uniform samples."""
  signal = np.random.uniform(minval, maxval, sample_shape)
  return ddsp.core.resample(signal, n_timesteps, method=method)


def normal_generator(sample_shape, n_timesteps, mean, stddev, method='linear'):
  """Linearly interpolates between a fixed number of uniform samples."""
  signal = np.random.normal(mean, stddev, sample_shape)
  return ddsp.core.resample(signal, n_timesteps, method=method)


def modulate(signal, maxval=0.5, n_t=10, method='linear'):
  """Generate abs(normal noise) with stddev from uniform, multiply original."""
  n_batch, n_timesteps, _ = signal.shape
  signal_std = np.random.uniform(0.0, maxval, n_batch)
  mod = np.abs(np.random.normal(1.0, signal_std, [1, n_t, 1]))
  mod = np.transpose(mod, [2, 1, 0])
  mod = ddsp.core.resample(mod, n_timesteps, method=method)
  return signal * mod


@gin.configurable
def generate_notes(n_batch,
                   n_timesteps,
                   n_harmonics=100,
                   n_mags=65,
                   get_controls=True):
  """Generate self-supervision signal of discrete notes."""
  n_notes = uniform_int(1, 20)

  # Amplitudes.
  method = 'nearest' if flip(0.5) else 'linear'
  harm_amp = uniform_generator([n_batch, n_notes, 1], n_timesteps,
                               minval=-2, maxval=2, method=method)
  if get_controls:
    harm_amp = ddsp.core.exp_sigmoid(harm_amp)

  # Frequencies.
  note_midi = uniform_generator([n_batch, n_notes, 1], n_timesteps,
                                minval=24.0, maxval=84.0, method='nearest')
  f0_hz = ddsp.core.midi_to_hz(note_midi)

  # Harmonic Distribution
  method = 'nearest' if flip(0.5) else 'linear'
  n_lines = 10
  exponents = [uniform_float(1.0, 6.0) for _ in range(n_lines)]
  harm_dist_lines = [-tf.linspace(0.0, float(i), n_harmonics)**exponents[i]
                     for i in range(n_lines)]
  harm_dist_lines = tf.stack(harm_dist_lines)
  lines_dist = uniform_generator([n_batch, n_notes, n_lines], n_timesteps,
                                 minval=0.0, maxval=1.0, method=method)
  harm_dist = (lines_dist[..., tf.newaxis] *
               harm_dist_lines[tf.newaxis, tf.newaxis, :])
  harm_dist = tf.reduce_sum(harm_dist, axis=-2)

  if get_controls:
    harm_dist = ddsp.core.exp_sigmoid(harm_dist)
    harm_dist = ddsp.core.remove_above_nyquist(f0_hz, harm_dist)
    harm_dist = ddsp.core.safe_divide(
        harm_dist, tf.reduce_sum(harm_dist, axis=-1, keepdims=True))

  # Noise Magnitudes.
  method = 'nearest' if flip(0.5) else 'linear'
  mags = uniform_generator([n_batch, n_notes, n_mags], n_timesteps,
                           minval=-6.0, maxval=uniform_float(-4.0, 0.0),
                           method=method)
  if get_controls:
    mags = ddsp.core.exp_sigmoid(mags)

  sin_amps, sin_freqs = ddsp.core.harmonic_to_sinusoidal(
      harm_amp, harm_dist, f0_hz)

  controls = {'harm_amp': harm_amp,
              'harm_dist': harm_dist,
              'f0_hz': f0_hz,
              'sin_amps': sin_amps,
              'sin_freqs': sin_freqs,
              'noise_magnitudes': mags}
  return controls


def random_blend(length, env_start=1.0, env_end=0.0, exp_max=2.0):
  """Returns a linear mix between two values, with a random curve steepness."""
  exp = uniform_float(-exp_max, exp_max)
  v = np.linspace(1.0, 0.0, length) ** (2.0 ** exp)
  return env_start * v + env_end * (1.0 - v)


def random_harm_dist(n_harmonics=100, low_pass=True, rand_phase=0.0):
  """Create harmonic distribution out of sinusoidal components."""
  n_components = uniform_int(1, 20)
  smoothness = uniform_float(1.0, 10.0)
  coeffs = np.random.rand(n_components)
  freqs = np.random.rand(n_components) * n_harmonics / smoothness

  v = []
  for i in range(n_components):
    v_i = (coeffs[i] * np.cos(
        np.linspace(0.0, 2.0 * np.pi * freqs[i], n_harmonics) +
        uniform_float(0.0, np.pi * 2.0 * rand_phase)))
    v.append(v_i)

  if low_pass:
    v = [v_i * np.linspace(1.0, uniform_float(0.0, 0.5), n_harmonics) **
         uniform_float(0.5, 2.0) for v_i in v]
  harm_dist = np.sum(np.stack(v), axis=0)
  return harm_dist


@gin.configurable
def generate_notes_v2(n_batch=1,
                      n_timesteps=125,
                      n_harmonics=100,
                      n_mags=65,
                      min_note_length=5,
                      max_note_length=25,
                      p_silent=0.1,
                      p_vibrato=0.5,
                      get_controls=True):
  """Generate more expressive self-supervision signal of discrete notes."""
  harm_amp = np.zeros([n_batch, n_timesteps, 1])
  harm_dist = np.zeros([n_batch, n_timesteps, n_harmonics])
  f0_midi = np.zeros([n_batch, n_timesteps, 1])
  mags = np.zeros([n_batch, n_timesteps, n_mags])

  for b in range(n_batch):
    t_start = 0
    while t_start < n_timesteps:
      note_length = uniform_int(min_note_length, max_note_length)
      t_end = min(t_start + note_length, n_timesteps)
      note_length = t_end - t_start

      # Silent?
      silent = flip(p_silent)
      if silent:
        # Amplitudes.
        ha_slice = harm_amp[b, t_start:t_end, :]
        ha_slice -= 10.0

      else:
        # Amplitudes.
        amp_start = uniform_float(-1.0, 3.0)
        amp_end = uniform_float(-1.0, 3.0)
        amp_blend = random_blend(note_length, amp_start, amp_end)
        ha_slice = harm_amp[b, t_start:t_end, :]
        ha_slice += amp_blend[:, np.newaxis]

        # Add some noise.
        ha_slice += uniform_float(0.0, 0.1) * np.random.randn(*ha_slice.shape)

        # Harmonic Distribution.
        low_pass = flip(0.8)
        rand_phase = uniform_float(0.0, 0.4)
        harm_dist_start = random_harm_dist(n_harmonics,
                                           low_pass=low_pass,
                                           rand_phase=rand_phase)[np.newaxis, :]
        harm_dist_end = random_harm_dist(n_harmonics,
                                         low_pass=low_pass,
                                         rand_phase=rand_phase)[np.newaxis, :]
        blend = random_blend(note_length, 1.0, 0.0)[:, np.newaxis]
        harm_dist_blend = (harm_dist_start * blend +
                           harm_dist_end * (1.0 - blend))
        hd_slice = harm_dist[b, t_start:t_end, :]
        hd_slice += harm_dist_blend

        # Add some noise.
        hd_slice += uniform_float(0.0, 0.5) * np.random.randn(*hd_slice.shape)

        # Fundamental Frequency.
        f0 = uniform_float(24.0, 84.0)
        if flip(p_vibrato):
          vib_start = uniform_float(0.0, 1.0)
          vib_end = uniform_float(0.0, 1.0)
          vib_periods = uniform_float(0.0, note_length * 2.0 / min_note_length)
          vib_blend = random_blend(note_length, vib_start, vib_end)
          vib = vib_blend * np.sin(
              np.linspace(0.0, 2.0 * np.pi * vib_periods, note_length))
          f0_note = f0 + vib
        else:
          f0_note = f0 * np.ones([note_length])

        f0_slice = f0_midi[b, t_start:t_end, :]
        f0_slice += f0_note[:, np.newaxis]

        # Add some noise.
        f0_slice += uniform_float(0.0, 0.1) * np.random.randn(*f0_slice.shape)

      # Filtered Noise.
      low_pass = flip(0.8)
      rand_phase = uniform_float(0.0, 0.4)
      mags_start = random_harm_dist(n_mags,
                                    low_pass=low_pass,
                                    rand_phase=rand_phase)[np.newaxis, :]
      mags_end = random_harm_dist(n_mags,
                                  low_pass=low_pass,
                                  rand_phase=rand_phase)[np.newaxis, :]
      blend = random_blend(note_length, 1.0, 0.0)[:, np.newaxis]
      mags_blend = mags_start * blend + mags_end * (1.0 - blend)

      mags_slice = mags[b, t_start:t_end, :]
      mags_slice += mags_blend

      # Add some noise.
      mags_slice += uniform_float(0.0, 0.2) * np.random.randn(*mags_slice.shape)

      # # Scale.
      mags_slice -= uniform_float(1.0, 10.0)

      t_start = t_end

  if get_controls:
    harm_amp = ddsp.core.exp_sigmoid(harm_amp)
    harm_amp /= uniform_float(1.0, [2.0, uniform_float(2.0, 10.0)][flip(0.2)])

  # Frequencies.
  f0_hz = ddsp.core.midi_to_hz(f0_midi)

  if get_controls:
    harm_dist = tf.nn.softmax(harm_dist)
    harm_dist = ddsp.core.remove_above_nyquist(f0_hz, harm_dist)
    harm_dist = ddsp.core.safe_divide(
        harm_dist, tf.reduce_sum(harm_dist, axis=-1, keepdims=True))

  if get_controls:
    mags = ddsp.core.exp_sigmoid(mags)

  sin_amps, sin_freqs = ddsp.core.harmonic_to_sinusoidal(
      harm_amp, harm_dist, f0_hz)

  controls = {'harm_amp': harm_amp,
              'harm_dist': harm_dist,
              'f0_hz': f0_hz,
              'sin_amps': sin_amps,
              'sin_freqs': sin_freqs,
              'noise_magnitudes': mags}
  return controls


