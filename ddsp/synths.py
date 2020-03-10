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
"""Library of synthesizer functions."""

from ddsp import core
from ddsp import processors
import gin
import tensorflow.compat.v2 as tf


@gin.register
class Additive(processors.Processor):
  """Synthesize audio with a bank of harmonic sinusoidal oscillators."""

  def __init__(self,
               n_samples=64000,
               sample_rate=16000,
               scale_fn=core.exp_sigmoid,
               normalize_below_nyquist=True,
               name='additive'):
    super().__init__(name=name)
    self.n_samples = n_samples
    self.sample_rate = sample_rate
    self.scale_fn = scale_fn
    self.normalize_below_nyquist = normalize_below_nyquist

  def get_controls(self,
                   amplitudes,
                   harmonic_distribution,
                   f0_hz):
    """Convert network output tensors into a dictionary of synthesizer controls.

    Args:
      amplitudes: 3-D Tensor of synthesizer controls, of shape
        [batch, time, 1].
      harmonic_distribution: 3-D Tensor of synthesizer controls, of shape
        [batch, time, n_harmonics].
      f0_hz: Fundamental frequencies in hertz. Shape [batch, time, 1].

    Returns:
      controls: Dictionary of tensors of synthesizer controls.
    """
    # Scale the amplitudes.
    if self.scale_fn is not None:
      amplitudes = self.scale_fn(amplitudes)
      harmonic_distribution = self.scale_fn(harmonic_distribution)

    # Bandlimit the harmonic distribution.
    if self.normalize_below_nyquist:
      n_harmonics = int(harmonic_distribution.shape[-1])
      harmonic_frequencies = core.get_harmonic_frequencies(f0_hz,
                                                           n_harmonics)
      harmonic_distribution = core.remove_above_nyquist(harmonic_frequencies,
                                                        harmonic_distribution,
                                                        self.sample_rate)

    # Normalize
    harmonic_distribution /= tf.reduce_sum(harmonic_distribution,
                                           axis=-1,
                                           keepdims=True)

    return {'amplitudes': amplitudes,
            'harmonic_distribution': harmonic_distribution,
            'f0_hz': f0_hz}

  def get_signal(self, amplitudes, harmonic_distribution, f0_hz):
    """Synthesize audio with additive synthesizer from controls.

    Args:
      amplitudes: Amplitude tensor of shape [batch, n_frames, 1]. Expects
        float32 that is strictly positive.
      harmonic_distribution: Tensor of shape [batch, n_frames, n_harmonics].
        Expects float32 that is strictly positive and normalized in the last
        dimension.
      f0_hz: The fundamental frequency in Hertz. Tensor of shape [batch,
        n_frames, 1].

    Returns:
      signal: A tensor of harmonic waves of shape [batch, n_samples].
    """
    signal = core.harmonic_synthesis(
        frequencies=f0_hz,
        amplitudes=amplitudes,
        harmonic_distribution=harmonic_distribution,
        n_samples=self.n_samples,
        sample_rate=self.sample_rate)
    return signal


@gin.register
class FilteredNoise(processors.Processor):
  """Synthesize audio by filtering white noise."""

  def __init__(self,
               n_samples=64000,
               window_size=257,
               scale_fn=core.exp_sigmoid,
               initial_bias=-5.0,
               name='filtered_noise'):
    super().__init__(name=name)
    self.n_samples = n_samples
    self.window_size = window_size
    self.scale_fn = scale_fn
    self.initial_bias = initial_bias

  def get_controls(self, magnitudes):
    """Convert network outputs into a dictionary of synthesizer controls.

    Args:
      magnitudes: 3-D Tensor of synthesizer parameters, of shape [batch, time,
        n_filter_banks].

    Returns:
      controls: Dictionary of tensors of synthesizer controls.
    """
    # Scale the magnitudes.
    if self.scale_fn is not None:
      magnitudes = self.scale_fn(magnitudes + self.initial_bias)

    return {'magnitudes': magnitudes}

  def get_signal(self, magnitudes):
    """Synthesize audio with filtered white noise.

    Args:
      magnitudes: Magnitudes tensor of shape [batch, n_frames, n_filter_banks].
        Expects float32 that is strictly positive.

    Returns:
      signal: A tensor of harmonic waves of shape [batch, n_samples, 1].
    """
    batch_size = int(magnitudes.shape[0])
    signal = tf.random.uniform(
        [batch_size, self.n_samples], minval=-1.0, maxval=1.0)
    return core.frequency_filter(signal,
                                 magnitudes,
                                 window_size=self.window_size)


@gin.register
class Wavetable(processors.Processor):
  """Synthesize audio from a series of wavetables."""

  def __init__(self,
               n_samples=64000,
               sample_rate=16000,
               scale_fn=core.exp_sigmoid,
               name='wavetable'):
    super().__init__(name=name)
    self.n_samples = n_samples
    self.sample_rate = sample_rate
    self.scale_fn = scale_fn

  def get_controls(self,
                   amplitudes,
                   wavetables,
                   f0_hz):
    """Convert network output tensors into a dictionary of synthesizer controls.

    Args:
      amplitudes: 3-D Tensor of synthesizer controls, of shape
        [batch, time, 1].
      wavetables: 3-D Tensor of synthesizer controls, of shape
        [batch, time, n_harmonics].
      f0_hz: Fundamental frequencies in hertz. Shape [batch, time, 1].

    Returns:
      controls: Dictionary of tensors of synthesizer controls.
    """
    # Scale the amplitudes.
    if self.scale_fn is not None:
      amplitudes = self.scale_fn(amplitudes)
      wavetables = self.scale_fn(wavetables)

    return  {'amplitudes': amplitudes,
             'wavetables': wavetables,
             'f0_hz': f0_hz}

  def get_signal(self, amplitudes, wavetables, f0_hz):
    """Synthesize audio with additive synthesizer from controls.

    Args:
      amplitudes: Amplitude tensor of shape [batch, n_frames, 1]. Expects
        float32 that is strictly positive.
      wavetables: Tensor of shape [batch, n_frames, n_wavetable].
      f0_hz: The fundamental frequency in Hertz. Tensor of shape [batch,
        n_frames, 1].

    Returns:
      signal: A tensor of of shape [batch, n_samples].
    """
    wavetables = core.resample(wavetables, self.n_samples)
    signal = core.wavetable_synthesis(amplitudes=amplitudes,
                                      wavetables=wavetables,
                                      frequencies=f0_hz,
                                      n_samples=self.n_samples,
                                      sample_rate=self.sample_rate)
    return signal


