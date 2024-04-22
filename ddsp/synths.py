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

"""Library of synthesizer functions."""

from ddsp import core
from ddsp import processors
import gin
import tensorflow.compat.v2 as tf


@gin.register
class TensorToAudio(processors.Processor):
  """Identity "synth" returning input samples with channel dimension removed."""

  def __init__(self, name='tensor_to_audio'):
    super().__init__(name=name)

  def get_controls(self, samples):
    """Convert network output tensors into a dictionary of synthesizer controls.

    Args:
      samples: 3-D Tensor of "controls" (really just samples), of shape
        [batch, time, 1].

    Returns:
      Dictionary of tensors of synthesizer controls.
    """
    return {'samples': samples}

  def get_signal(self, samples):
    """"Synthesize" audio by removing channel dimension from input samples.

    Args:
      samples: 3-D Tensor of "controls" (really just samples), of shape
        [batch, time, 1].

    Returns:
      A tensor of audio with shape [batch, time].
    """
    return tf.squeeze(samples, 2)


@gin.register
class Harmonic(processors.Processor):
  """Synthesize audio with a bank of harmonic sinusoidal oscillators."""

  def __init__(self,
               n_samples=64000,
               sample_rate=16000,
               scale_fn=core.exp_sigmoid,
               normalize_below_nyquist=True,
               amp_resample_method='window',
               use_angular_cumsum=False,
               name='harmonic'):
    """Constructor.

    Args:
      n_samples: Fixed length of output audio.
      sample_rate: Samples per a second.
      scale_fn: Scale function for amplitude and harmonic distribution inputs.
      normalize_below_nyquist: Remove harmonics above the nyquist frequency
        and normalize the remaining harmonic distribution to sum to 1.0.
      amp_resample_method: Mode with which to resample amplitude envelopes.
        Must be in ['nearest', 'linear', 'cubic', 'window']. 'window' uses
        overlapping windows (only for upsampling) which is smoother
        for amplitude envelopes with large frame sizes.
      use_angular_cumsum: Use angular cumulative sum on accumulating phase
        instead of tf.cumsum. If synthesized examples are longer than ~100k
        audio samples, consider use_angular_cumsum to avoid accumulating
        noticible phase errors due to the limited precision of tf.cumsum.
        However, using angular cumulative sum is slower on accelerators.
      name: Synth name.
    """
    super().__init__(name=name)
    self.n_samples = n_samples
    self.sample_rate = sample_rate
    self.scale_fn = scale_fn
    self.normalize_below_nyquist = normalize_below_nyquist
    self.amp_resample_method = amp_resample_method
    self.use_angular_cumsum = use_angular_cumsum

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

    harmonic_distribution = core.normalize_harmonics(
        harmonic_distribution, f0_hz,
        self.sample_rate if self.normalize_below_nyquist else None)

    return {'amplitudes': amplitudes,
            'harmonic_distribution': harmonic_distribution,
            'f0_hz': f0_hz}

  def get_signal(self, amplitudes, harmonic_distribution, f0_hz):
    """Synthesize audio with harmonic synthesizer from controls.

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
        sample_rate=self.sample_rate,
        amp_resample_method=self.amp_resample_method,
        use_angular_cumsum=self.use_angular_cumsum)
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
    """Synthesize audio with wavetable synthesizer from controls.

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


@gin.register
class Sinusoidal(processors.Processor):
  """Synthesize audio with a bank of arbitrary sinusoidal oscillators."""

  def __init__(self,
               n_samples=64000,
               sample_rate=16000,
               amp_scale_fn=core.exp_sigmoid,
               amp_resample_method='window',
               freq_scale_fn=core.frequencies_sigmoid,
               name='sinusoidal'):
    super().__init__(name=name)
    self.n_samples = n_samples
    self.sample_rate = sample_rate
    self.amp_scale_fn = amp_scale_fn
    self.amp_resample_method = amp_resample_method
    self.freq_scale_fn = freq_scale_fn

  def get_controls(self, amplitudes, frequencies):
    """Convert network output tensors into a dictionary of synthesizer controls.

    Args:
      amplitudes: 3-D Tensor of synthesizer controls, of shape
        [batch, time, n_sinusoids].
      frequencies: 3-D Tensor of synthesizer controls, of shape
        [batch, time, n_sinusoids]. Expects strictly positive in Hertz.

    Returns:
      controls: Dictionary of tensors of synthesizer controls.
    """
    # Scale the inputs.
    if self.amp_scale_fn is not None:
      amplitudes = self.amp_scale_fn(amplitudes)

    if self.freq_scale_fn is not None:
      frequencies = self.freq_scale_fn(frequencies)
      amplitudes = core.remove_above_nyquist(frequencies,
                                             amplitudes,
                                             self.sample_rate)

    return {'amplitudes': amplitudes,
            'frequencies': frequencies}

  def get_signal(self, amplitudes, frequencies):
    """Synthesize audio with sinusoidal synthesizer from controls.

    Args:
      amplitudes: Amplitude tensor of shape [batch, n_frames, n_sinusoids].
        Expects float32 that is strictly positive.
      frequencies: Tensor of shape [batch, n_frames, n_sinusoids].
        Expects float32 in Hertz that is strictly positive.

    Returns:
      signal: A tensor of harmonic waves of shape [batch, n_samples].
    """
    # Create sample-wise envelopes.
    amplitude_envelopes = core.resample(amplitudes, self.n_samples,
                                        method=self.amp_resample_method)
    frequency_envelopes = core.resample(frequencies, self.n_samples)

    signal = core.oscillator_bank(frequency_envelopes=frequency_envelopes,
                                  amplitude_envelopes=amplitude_envelopes,
                                  sample_rate=self.sample_rate)
    return signal


