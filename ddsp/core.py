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
"""Library of functions for differentiable digital signal processing (DDSP)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from typing import Any, Dict, Text, TypeVar

import gin
import numpy as np
from scipy import fftpack
import tensorflow.compat.v2 as tf

Number = TypeVar('Number', int, float, np.ndarray, tf.Tensor)


# Utility Functions ------------------------------------------------------------
def tf_float32(x):
  """Ensure array/tensor is a float32 tf.Tensor."""
  if isinstance(x, tf.Tensor):
    return tf.cast(x, dtype=tf.float32)  # This is a no-op if x is float32.
  else:
    return tf.convert_to_tensor(x, tf.float32)


def make_iterable(x):
  """Ensure that x is an iterable."""
  return x if isinstance(x, collections.Iterable) else [x]


def nested_lookup(nested_key: Text,
                  nested_dict: Dict[Text, Any],
                  delimiter: Text = '/') -> tf.Tensor:
  """Returns the value of a nested dict according to a parsed input string.

  Args:
    nested_key: String of the form "key/key/key...".
    nested_dict: Nested dictionary.
    delimiter: String that splits the nested keys.

  Returns:
    value: Value of the key from the nested dictionary.
  """
  # Parse the input string.
  keys = nested_key.split(delimiter)
  # Return the nested value.
  value = nested_dict
  for key in keys:
    value = value[key]
  return value


def midi_to_hz(notes: Number) -> Number:
  """TF-compatible midi_to_hz function."""
  notes = tf_float32(notes)
  return 440.0 * (2.0**((notes - 69.0) / 12.0))


def hz_to_midi(frequencies: Number) -> Number:
  """TF-compatible hz_to_midi function."""
  frequencies = tf_float32(frequencies)
  log2 = lambda x: tf.math.log(x) / tf.math.log(2.0)
  notes = 12.0 * (log2(frequencies) - log2(440.0)) + 69.0
  # Map 0 Hz to MIDI 0 (Replace -inf with 0.)
  cond = tf.equal(notes, -np.inf)
  notes = tf.where(cond, 0.0, notes)
  return notes


def unit_to_midi(unit: Number,
                 midi_min: Number = 20.0,
                 midi_max: Number = 90.0,
                 clip: bool = False) -> Number:
  """Map the unit interval [0, 1] to MIDI notes."""
  unit = tf.clip_by_value(unit, 0.0, 1.0) if clip else unit
  return midi_min + (midi_max - midi_min) * unit


def midi_to_unit(midi: Number,
                 midi_min: Number = 20.0,
                 midi_max: Number = 90.0,
                 clip: bool = False) -> Number:
  """Map MIDI notes to the unit interval [0, 1]."""
  unit = (midi - midi_min) / (midi_max - midi_min)
  return tf.clip_by_value(unit, 0.0, 1.0) if clip else unit


def unit_to_hz(unit: Number,
               hz_min: Number,
               hz_max: Number,
               clip: bool = False) -> Number:
  """Map unit interval [0, 1] to [hz_min, hz_max], scaling logarithmically."""
  midi = unit_to_midi(unit,
                      midi_min=hz_to_midi(hz_min),
                      midi_max=hz_to_midi(hz_max),
                      clip=clip)
  return midi_to_hz(midi)


def hz_to_unit(hz: Number,
               hz_min: Number,
               hz_max: Number,
               clip: bool = False) -> Number:
  """Map [hz_min, hz_max] to unit interval [0, 1], scaling logarithmically."""
  midi = hz_to_midi(hz)
  return midi_to_unit(midi,
                      midi_min=hz_to_midi(hz_min),
                      midi_max=hz_to_midi(hz_max),
                      clip=clip)


def resample(inputs: tf.Tensor,
             n_timesteps: int,
             method: Text = 'linear',
             add_endpoint: bool = True) -> tf.Tensor:
  """Interpolates a tensor from n_frames to n_timesteps.

  Args:
    inputs: Framewise 1-D, 2-D, 3-D, or 4-D Tensor. Shape [n_frames],
      [batch_size, n_frames], [batch_size, n_frames, channels], or
      [batch_size, n_frames, n_freq, channels].
    n_timesteps: Time resolution of the output signal.
    method: Type of resampling, must be in ['linear', 'cubic', 'window']. Linear
      and cubic ar typical bilinear, bicubic interpolation. Window uses
      overlapping windows (only for upsampling) which is smoother for amplitude
      envelopes.
    add_endpoint: Hold the last timestep for an additional step as the endpoint.
      Then, n_timesteps is divided evenly into n_frames segments. If false, use
      the last timestep as the endpoint, producing (n_frames - 1) segments with
      each having a length of n_timesteps / (n_frames - 1).
  Returns:
    Interpolated 1-D, 2-D, 3-D, or 4-D Tensor. Shape [n_timesteps],
      [batch_size, n_timesteps], [batch_size, n_timesteps, channels], or
      [batch_size, n_timesteps, n_freqs, channels].

  Raises:
    ValueError: If method is 'window' and input is 4-D.
    ValueError: If method is not one of 'linear', 'cubic', or 'window'.
  """
  inputs = tf_float32(inputs)
  is_1d = len(inputs.shape) == 1
  is_2d = len(inputs.shape) == 2
  is_4d = len(inputs.shape) == 4

  # Ensure inputs are at least 3d.
  if is_1d:
    inputs = inputs[tf.newaxis, :, tf.newaxis]
  elif is_2d:
    inputs = inputs[:, :, tf.newaxis]

  def _image_resize(method):
    """Closure around tf.image.resize."""
    # Image resize needs 4-D input. Add/remove extra axis if not 4-D.
    outputs = inputs[:, :, tf.newaxis, :] if not is_4d else inputs
    outputs = tf.compat.v1.image.resize(outputs,
                                        [n_timesteps, outputs.shape[2]],
                                        method=method,
                                        align_corners=not add_endpoint)
    return outputs[:, :, 0, :] if not is_4d else outputs

  # Perform resampling.
  if method == 'linear':
    outputs = _image_resize(tf.compat.v1.image.ResizeMethod.BILINEAR)
  elif method == 'cubic':
    outputs = _image_resize(tf.compat.v1.image.ResizeMethod.BICUBIC)
  elif method == 'window':
    outputs = upsample_with_windows(inputs, n_timesteps, add_endpoint)
  else:
    raise ValueError('Method ({}) is invalid. Must be one of {}.'.format(
        method, "['linear', 'cubic', 'window']"))

  # Return outputs to the same dimensionality of the inputs.
  if is_1d:
    outputs = outputs[0, :, 0]
  elif is_2d:
    outputs = outputs[:, :, 0]

  return outputs


def upsample_with_windows(inputs: tf.Tensor,
                          n_timesteps: int,
                          add_endpoint: bool = True) -> tf.Tensor:
  """Upsample a series of frames using using overlapping hann windows.

  Good for amplitude envelopes.
  Args:
    inputs: Framewise 3-D tensor. Shape [batch_size, n_frames, n_channels].
    n_timesteps: The time resolution of the output signal.
    add_endpoint: Hold the last timestep for an additional step as the endpoint.
      Then, n_timesteps is divided evenly into n_frames segments. If false, use
      the last timestep as the endpoint, producing (n_frames - 1) segments with
      each having a length of n_timesteps / (n_frames - 1).

  Returns:
    Upsampled 3-D tensor. Shape [batch_size, n_timesteps, n_channels].

  Raises:
    ValueError: If input does not have 3 dimensions.
    ValueError: If attempting to use function for downsampling.
    ValueError: If n_timesteps is not divisible by n_frames (if add_endpoint is
      true) or n_frames - 1 (if add_endpoint is false).
  """
  inputs = tf_float32(inputs)

  if len(inputs.shape) != 3:
    raise ValueError('Upsample_with_windows() only supports 3 dimensions, '
                     'not {}.'.format(inputs.shape))

  # Mimic behavior of tf.image.resize.
  # For forward (not endpointed), hold value for last interval.
  if add_endpoint:
    inputs = tf.concat([inputs, inputs[:, -1:, :]], axis=1)

  n_frames = int(inputs.shape[1])
  n_intervals = (n_frames - 1)

  if n_frames >= n_timesteps:
    raise ValueError('Upsample with windows cannot be used for downsampling'
                     'More input frames ({}) than output timesteps ({})'.format(
                         n_frames, n_timesteps))

  if n_timesteps % n_intervals != 0.0:
    minus_one = '' if add_endpoint else ' - 1'
    raise ValueError(
        'For upsampling, the target the number of timesteps must be divisible '
        'by the number of input frames{}. (timesteps:{}, frames:{}, '
        'add_endpoint={}).'.format(minus_one, n_timesteps, n_frames,
                                   add_endpoint))

  # Constant overlap-add, half overlapping windows.
  hop_size = n_timesteps // n_intervals
  window_length = 2 * hop_size
  window = tf.signal.hann_window(window_length)  # [window]

  # Transpose for overlap_and_add.
  x = tf.transpose(inputs, perm=[0, 2, 1])  # [batch_size, n_channels, n_frames]

  # Broadcast multiply.
  # Add dimension for windows [batch_size, n_channels, n_frames, window].
  x = x[:, :, :, tf.newaxis]
  window = window[tf.newaxis, tf.newaxis, tf.newaxis, :]
  x_windowed = (x * window)
  x = tf.signal.overlap_and_add(x_windowed, hop_size)

  # Transpose back.
  x = tf.transpose(x, perm=[0, 2, 1])  # [batch_size, n_timesteps, n_channels]

  # Trim the rise and fall of the first and last window.
  return x[:, hop_size:-hop_size, :]


def log_scale(x, min_x, max_x):
  """Scales a -1 to 1 value logarithmically between min and max."""
  x = tf_float32(x)
  x = (x + 1.0) / 2.0  # Scale [-1, 1] to [0, 1]
  return tf.exp((1.0 - x) * tf.math.log(min_x) + x * tf.math.log(max_x))


@gin.register
def exp_sigmoid(x, exponent=10.0, max_value=2.0, threshold=1e-7):
  """Exponentiated Sigmoid pointwise nonlinearity.

  Bounds input to [threshold, max_value] with slope given by exponent.

  Args:
    x: Input tensor.
    exponent: In nonlinear regime (away from x=0), the output varies by this
      factor for every change of x by 1.0.
    max_value: Limiting value at x=inf.
    threshold: Limiting value at x=-inf. Stablizes training when outputs are
      pushed to 0.

  Returns:
    A tensor with pointwise nonlinearity applied.
  """
  x = tf_float32(x)
  return max_value * tf.nn.sigmoid(x)**tf.math.log(exponent) + threshold


@gin.register
def sym_exp_sigmoid(x, width=8.0):
  """Symmetrical version of exp_sigmoid centered at (0, 1e-7)."""
  x = tf_float32(x)
  return exp_sigmoid(width * (tf.abs(x)/2.0 - 1.0))


# Additive Synthesizer ---------------------------------------------------------
def remove_above_nyquist(frequency_envelopes: tf.Tensor,
                         amplitude_envelopes: tf.Tensor,
                         sample_rate: int = 16000) -> tf.Tensor:
  """Set amplitudes for oscillators above nyquist to 0.

  Args:
    frequency_envelopes: Sample-wise oscillator frequencies (Hz). Shape
      [batch_size, n_samples, n_sinusoids].
    amplitude_envelopes: Sample-wise oscillator amplitude. Shape [batch_size,
      n_samples, n_sinusoids].
    sample_rate: Sample rate in samples per a second.

  Returns:
    amplitude_envelopes: Sample-wise filtered oscillator amplitude.
      Shape [batch_size, n_samples, n_sinusoids].
  """
  frequency_envelopes = tf_float32(frequency_envelopes)
  amplitude_envelopes = tf_float32(amplitude_envelopes)

  amplitude_envelopes = tf.where(
      tf.greater_equal(frequency_envelopes, sample_rate / 2.0),
      tf.zeros_like(amplitude_envelopes), amplitude_envelopes)
  return amplitude_envelopes


def oscillator_bank(frequency_envelopes: tf.Tensor,
                    amplitude_envelopes: tf.Tensor,
                    sample_rate: int = 16000,
                    sum_sinusoids: bool = True) -> tf.Tensor:
  """Generates audio from sample-wise frequencies for a bank of oscillators.

  Args:
    frequency_envelopes: Sample-wise oscillator frequencies (Hz). Shape
      [batch_size, n_samples, n_sinusoids].
    amplitude_envelopes: Sample-wise oscillator amplitude. Shape [batch_size,
      n_samples, n_sinusoids].
    sample_rate: Sample rate in samples per a second.
    sum_sinusoids: Add up audio from all the sinusoids.

  Returns:
    wav: Sample-wise audio. Shape [batch_size, n_samples, n_sinusoids] if
      sum_sinusoids=False, else shape is [batch_size, n_samples].
  """
  frequency_envelopes = tf_float32(frequency_envelopes)
  amplitude_envelopes = tf_float32(amplitude_envelopes)

  # Don't exceed Nyquist.
  amplitude_envelopes = remove_above_nyquist(frequency_envelopes,
                                             amplitude_envelopes,
                                             sample_rate)

  # Change Hz to radians per sample.
  omegas = frequency_envelopes * (2.0 * np.pi)  # rad / sec
  omegas = omegas / float(sample_rate)  # rad / sample

  # Accumulate phase and synthesize.
  phases = tf.cumsum(omegas, axis=1)
  wavs = tf.sin(phases)
  audio = amplitude_envelopes * wavs  # [mb, n_samples, n_sinusoids]
  if sum_sinusoids:
    audio = tf.reduce_sum(audio, axis=-1)  # [mb, n_samples]
  return audio


def get_harmonic_frequencies(frequencies: tf.Tensor,
                             n_harmonics: int) -> tf.Tensor:
  """Create integer multiples of the fundamental frequency.

  Args:
    frequencies: Fundamental frequencies (Hz). Shape [batch_size, :, 1].
    n_harmonics: Number of harmonics.

  Returns:
    harmonic_frequencies: Oscillator frequencies (Hz).
      Shape [batch_size, :, n_harmonics].
  """
  frequencies = tf_float32(frequencies)

  f_ratios = tf.linspace(1.0, float(n_harmonics), int(n_harmonics))
  f_ratios = f_ratios[tf.newaxis, tf.newaxis, :]
  harmonic_frequencies = frequencies * f_ratios
  return harmonic_frequencies


def harmonic_synthesis(frequencies: tf.Tensor,
                       amplitudes: tf.Tensor,
                       harmonic_shifts: tf.Tensor = None,
                       harmonic_distribution: tf.Tensor = None,
                       n_samples: int = 64000,
                       sample_rate: int = 16000,
                       amp_resample_method: Text = 'window') -> tf.Tensor:
  """Generate audio from frame-wise monophonic harmonic oscillator bank.

  Args:
    frequencies: Frame-wise fundamental frequency in Hz. Shape [batch_size,
      n_frames, 1].
    amplitudes: Frame-wise oscillator peak amplitude. Shape [batch_size,
      n_frames, 1].
    harmonic_shifts: Harmonic frequency variations (Hz), zero-centered. Total
      frequency of a harmonic is equal to (frequencies * harmonic_number * (1 +
      harmonic_shifts)). Shape [batch_size, n_frames, n_harmonics].
    harmonic_distribution: Harmonic amplitude variations, ranged zero to one.
      Total amplitude of a harmonic is equal to (amplitudes *
      harmonic_distribution). Shape [batch_size, n_frames, n_harmonics].
    n_samples: Total length of output audio. Interpolates and crops to this.
    sample_rate: Sample rate.
    amp_resample_method: Mode with which to resample amplitude envelopes.

  Returns:
    audio: Output audio. Shape [batch_size, n_samples, 1]
  """
  frequencies = tf_float32(frequencies)
  amplitudes = tf_float32(amplitudes)

  if harmonic_distribution is not None:
    harmonic_distribution = tf_float32(harmonic_distribution)
    n_harmonics = int(harmonic_distribution.shape[-1])
  elif harmonic_shifts is not None:
    harmonic_shifts = tf_float32(harmonic_shifts)
    n_harmonics = int(harmonic_shifts.shape[-1])
  else:
    n_harmonics = 1

  # Create harmonic frequencies [batch_size, n_frames, n_harmonics].
  harmonic_frequencies = get_harmonic_frequencies(frequencies, n_harmonics)
  if harmonic_shifts is not None:
    harmonic_frequencies *= (1.0 + harmonic_shifts)

  # Create harmonic amplitudes [batch_size, n_frames, n_harmonics].
  if harmonic_distribution is not None:
    harmonic_amplitudes = amplitudes * harmonic_distribution
  else:
    harmonic_amplitudes = amplitudes

  # Create sample-wise envelopes.
  frequency_envelopes = resample(harmonic_frequencies, n_samples)  # cycles/sec
  amplitude_envelopes = resample(harmonic_amplitudes, n_samples,
                                 method=amp_resample_method)

  # Synthesize from harmonics [batch_size, n_samples].
  audio = oscillator_bank(frequency_envelopes,
                          amplitude_envelopes,
                          sample_rate=sample_rate)
  return audio


# Wavetable Synthesizer --------------------------------------------------------
def linear_lookup(phase: tf.Tensor,
                  wavetables: tf.Tensor) -> tf.Tensor:
  """Lookup from wavetables with linear interpolation.

  Args:
    phase: The instantaneous phase of the base oscillator, ranging from 0 to
      1.0. This gives the position to lookup in the wavetable.
      Shape [batch_size, n_samples, 1].
    wavetables: Wavetables to be read from on lookup. Shape [batch_size,
      n_samples, n_wavetable] or [batch_size, n_wavetable].

  Returns:
    The resulting audio from linearly interpolated lookup of the wavetables at
      each point in time. Shape [batch_size, n_samples].
  """
  phase, wavetables = tf_float32(phase), tf_float32(wavetables)

  # Add a time dimension if not present.
  if len(wavetables.shape) == 2:
    wavetables = wavetables[:, tf.newaxis, :]

  # Add a wavetable dimension if not present.
  if len(phase.shape) == 2:
    phase = phase[:, :, tf.newaxis]

  # Add first sample to end of wavetable for smooth linear interpolation
  # between the last point in the wavetable and the first point.
  wavetables = tf.concat([wavetables, wavetables[..., 0:1]], axis=-1)
  n_wavetable = int(wavetables.shape[-1])

  # Get a phase value for each point on the wavetable.
  phase_wavetables = tf.linspace(0.0, 1.0, n_wavetable)

  # Get pair-wise distances from the oscillator phase to each wavetable point.
  # Axes are [batch, time, n_wavetable].
  phase_distance = tf.abs((phase - phase_wavetables[tf.newaxis, tf.newaxis, :]))

  # Put distance in units of wavetable samples.
  phase_distance *= n_wavetable - 1

  # Weighting for interpolation.
  # Distance is > 1.0 (and thus weights are 0.0) for all but nearest neighbors.
  weights = tf.nn.relu(1.0 - phase_distance)
  weighted_wavetables = weights * wavetables

  # Interpolated audio from summing the weighted wavetable at each timestep.
  return tf.reduce_sum(weighted_wavetables, axis=-1)


def wavetable_synthesis(frequencies: tf.Tensor,
                        amplitudes: tf.Tensor,
                        wavetables: tf.Tensor,
                        n_samples: int = 64000,
                        sample_rate: int = 16000):
  """Monophonic wavetable synthesizer.

  Args:
    frequencies: Frame-wise frequency in Hertz of the fundamental oscillator.
      Shape [batch_size, n_frames, 1].
    amplitudes: Frame-wise amplitude envelope to apply to the oscillator. Shape
      [batch_size, n_frames, 1].
    wavetables: Frame-wise wavetables from which to lookup. Shape
      [batch_size, n_wavetable] or [batch_size, n_frames, n_wavetable].
    n_samples: Total length of output audio. Interpolates and crops to this.
    sample_rate: Number of samples per a second.

  Returns:
    audio: Audio at the frequency and amplitude of the inputs, with harmonics
      given by the wavetable. Shape [batch_size, n_samples].
  """
  wavetables = tf_float32(wavetables)

  # Create sample-wise envelopes.
  amplitude_envelope = resample(amplitudes, n_samples, method='window')[:, :, 0]
  frequency_envelope = resample(frequencies, n_samples)  # cycles / sec

  # Create intermediate wavetables.
  wavetable_shape = wavetables.shape.as_list()
  if len(wavetable_shape) == 3 and wavetable_shape[1] > 1:
    wavetables = resample(wavetables, n_samples)

  # Accumulate phase (in cycles which range from 0.0 to 1.0).
  phase_velocity = frequency_envelope / float(sample_rate)  # cycles / sample

  # Note: Cumsum accumulates _very_ small errors at float32 precision.
  # On the order of milli-Hertz.
  phase = tf.cumsum(phase_velocity, axis=1, exclusive=True) % 1.0

  # Synthesize with linear lookup.
  audio = linear_lookup(phase, wavetables)

  # Modulate with amplitude envelope.
  audio *= amplitude_envelope
  return audio


def variable_length_delay(phase: tf.Tensor,
                          audio: tf.Tensor,
                          max_length: int = 512) -> tf.Tensor:
  """Delay audio by a time-vaying amount using linear interpolation.

  Useful for modulation effects such as vibrato, chorus, and flanging.
  Args:
    phase: The normlaized instantaneous length of the delay, ranging from 0 to
      1.0. This corresponds to a delay of 0 to max_length samples. Shape
      [batch_size, n_samples, 1].
    audio: Audio signal to be delayed. Shape [batch_size, n_samples].
    max_length: Maximimum delay in samples.

  Returns:
    The delayed audio signal. Shape [batch_size, n_samples].
  """
  phase, audio = tf_float32(phase), tf_float32(audio)

  # Make causal by zero-padding audio up front.
  audio = tf.pad(audio, [(0, 0), (max_length - 1, 0)])
  # Cut audio up into frames of max_length.
  frames = tf.signal.frame(audio,
                           frame_length=max_length,
                           frame_step=1,
                           pad_end=False)
  # Reverse frames so that [0, 1] phase corresponds to [0, max_length] delay.
  frames = frames[..., ::-1]
  # Read audio from the past frames.
  return linear_lookup(phase, frames)


# Time-varying convolution -----------------------------------------------------
def get_fft_size(frame_size: int, ir_size: int, power_of_2: bool = True) -> int:
  """Calculate final size for efficient FFT.

  Args:
    frame_size: Size of the audio frame.
    ir_size: Size of the convolving impulse response.
    power_of_2: Constrain to be a power of 2. If False, allow other 5-smooth
      numbers. TPU requires power of 2, while GPU is more flexible.

  Returns:
    fft_size: Size for efficient FFT.
  """
  convolved_frame_size = ir_size + frame_size - 1
  if power_of_2:
    # Next power of 2.
    fft_size = int(2**np.ceil(np.log2(convolved_frame_size)))
  else:
    fft_size = int(fftpack.helper.next_fast_len(convolved_frame_size))
  return fft_size


def crop_and_compensate_delay(audio: tf.Tensor, audio_size: int, ir_size: int,
                              padding: Text,
                              delay_compensation: int) -> tf.Tensor:
  """Crop audio output from convolution to compensate for group delay.

  Args:
    audio: Audio after convolution. Tensor of shape [batch, time_steps].
    audio_size: Initial size of the audio before convolution.
    ir_size: Size of the convolving impulse response.
    padding: Either 'valid' or 'same'. For 'same' the final output to be the
      same size as the input audio (audio_timesteps). For 'valid' the audio is
      extended to include the tail of the impulse response (audio_timesteps +
      ir_timesteps - 1).
    delay_compensation: Samples to crop from start of output audio to compensate
      for group delay of the impulse response. If delay_compensation < 0 it
      defaults to automatically calculating a constant group delay of the
      windowed linear phase filter from frequency_impulse_response().

  Returns:
    Tensor of cropped and shifted audio.

  Raises:
    ValueError: If padding is not either 'valid' or 'same'.
  """
  # Crop the output.
  if padding == 'valid':
    crop_size = ir_size + audio_size - 1
  elif padding == 'same':
    crop_size = audio_size
  else:
    raise ValueError('Padding must be \'valid\' or \'same\', instead '
                     'of {}.'.format(padding))

  # Compensate for the group delay of the filter by trimming the front.
  # For an impulse response produced by frequency_impulse_response(),
  # the group delay is constant because the filter is linear phase.
  total_size = int(audio.shape[-1])
  crop = total_size - crop_size
  start = ((ir_size - 1) // 2 -
           1 if delay_compensation < 0 else delay_compensation)
  end = crop - start
  return audio[:, start:-end]


def fft_convolve(audio: tf.Tensor,
                 impulse_response: tf.Tensor,
                 padding: Text = 'same',
                 delay_compensation: int = -1) -> tf.Tensor:
  """Filter audio with frames of time-varying impulse responses.

  Time-varying filter. Given audio [batch, n_samples], and a series of impulse
  responses [batch, n_frames, n_impulse_response], splits the audio into frames,
  applies filters, and then overlap-and-adds audio back together.
  Applies non-windowed non-overlapping STFT/ISTFT to efficiently compute
  convolution for large impulse response sizes.

  Args:
    audio: Input audio. Tensor of shape [batch, audio_timesteps].
    impulse_response: Finite impulse response to convolve. Can either be a 2-D
      Tensor of shape [batch, ir_size], or a 3-D Tensor of shape [batch,
      ir_frames, ir_size]. A 2-D tensor will apply a single linear
      time-invariant filter to the audio. A 3-D Tensor will apply a linear
      time-varying filter. Automatically chops the audio into equally shaped
      blocks to match ir_frames.
    padding: Either 'valid' or 'same'. For 'same' the final output to be the
      same size as the input audio (audio_timesteps). For 'valid' the audio is
      extended to include the tail of the impulse response (audio_timesteps +
      ir_timesteps - 1).
    delay_compensation: Samples to crop from start of output audio to compensate
      for group delay of the impulse response. If delay_compensation is less
      than 0 it defaults to automatically calculating a constant group delay of
      the windowed linear phase filter from frequency_impulse_response().

  Returns:
    audio_out: Convolved audio. Tensor of shape
        [batch, audio_timesteps + ir_timesteps - 1] ('valid' padding) or shape
        [batch, audio_timesteps] ('same' padding).

  Raises:
    ValueError: If audio and impulse response have different batch size.
    ValueError: If audio cannot be split into evenly spaced frames. (i.e. the
      number of impulse response frames is on the order of the audio size and
      not a multiple of the audio size.)
  """
  audio, impulse_response = tf_float32(audio), tf_float32(impulse_response)

  # Add a frame dimension to impulse response if it doesn't have one.
  ir_shape = impulse_response.shape.as_list()
  if len(ir_shape) == 2:
    impulse_response = impulse_response[:, tf.newaxis, :]
    ir_shape = impulse_response.shape.as_list()

  # Get shapes of audio and impulse response.
  batch_size_ir, n_ir_frames, ir_size = ir_shape
  batch_size, audio_size = audio.shape.as_list()

  # Validate that batch sizes match.
  if batch_size != batch_size_ir:
    raise ValueError('Batch size of audio ({}) and impulse response ({}) must '
                     'be the same.'.format(batch_size, batch_size_ir))

  # Cut audio into frames.
  frame_size = int(np.ceil(audio_size / n_ir_frames))
  hop_size = frame_size
  audio_frames = tf.signal.frame(audio, frame_size, hop_size, pad_end=True)

  # Check that number of frames match.
  n_audio_frames = int(audio_frames.shape[1])
  if n_audio_frames != n_ir_frames:
    raise ValueError(
        'Number of Audio frames ({}) and impulse response frames ({}) do not '
        'match. For small hop size = ceil(audio_size / n_ir_frames), '
        'number of impulse response frames must be a multiple of the audio '
        'size.'.format(n_audio_frames, n_ir_frames))

  # Pad and FFT the audio and impulse responses.
  fft_size = get_fft_size(frame_size, ir_size, power_of_2=True)
  audio_fft = tf.signal.rfft(audio_frames, [fft_size])
  ir_fft = tf.signal.rfft(impulse_response, [fft_size])

  # Multiply the FFTs (same as convolution in time).
  audio_ir_fft = tf.multiply(audio_fft, ir_fft)

  # Take the IFFT to resynthesize audio.
  audio_frames_out = tf.signal.irfft(audio_ir_fft)
  audio_out = tf.signal.overlap_and_add(audio_frames_out, hop_size)

  # Crop and shift the output audio.
  return crop_and_compensate_delay(audio_out, audio_size, ir_size, padding,
                                   delay_compensation)


# Filter Design ----------------------------------------------------------------
def apply_window_to_impulse_response(impulse_response: tf.Tensor,
                                     window_size: int = 0,
                                     causal: bool = False) -> tf.Tensor:
  """Apply a window to an impulse response and put in causal form.

  Args:
    impulse_response: A series of impulse responses frames to window, of shape
      [batch, n_frames, ir_size].
    window_size: Size of the window to apply in the time domain. If window_size
      is less than 1, it defaults to the impulse_response size.
    causal: Impulse responnse input is in causal form (peak in the middle).

  Returns:
    impulse_response: Windowed impulse response in causal form, with last
      dimension cropped to window_size if window_size is greater than 0 and less
      than ir_size.
  """
  impulse_response = tf_float32(impulse_response)

  # If IR is in causal form, put it in zero-phase form.
  if causal:
    impulse_response = tf.signal.fftshift(impulse_response, axes=-1)

  # Get a window for better time/frequency resolution than rectangular.
  # Window defaults to IR size, cannot be bigger.
  ir_size = int(impulse_response.shape[-1])
  if (window_size <= 0) or (window_size > ir_size):
    window_size = ir_size
  window = tf.signal.hann_window(window_size)

  # Zero pad the window and put in in zero-phase form.
  padding = ir_size - window_size
  if padding > 0:
    half_idx = (window_size + 1) // 2
    window = tf.concat([window[half_idx:],
                        tf.zeros([padding]),
                        window[:half_idx]], axis=0)
  else:
    window = tf.signal.fftshift(window, axes=-1)

  # Apply the window, to get new IR (both in zero-phase form).
  window = tf.broadcast_to(window, impulse_response.shape)
  impulse_response = window * tf.math.real(impulse_response)

  # Put IR in causal form and trim zero padding.
  if padding > 0:
    first_half_start = (ir_size - (half_idx - 1)) + 1
    second_half_end = half_idx + 1
    impulse_response = tf.concat([impulse_response[..., first_half_start:],
                                  impulse_response[..., :second_half_end]],
                                 axis=-1)
  else:
    impulse_response = tf.signal.fftshift(impulse_response, axes=-1)

  return impulse_response


def frequency_impulse_response(magnitudes: tf.Tensor,
                               window_size: int = 0) -> tf.Tensor:
  """Get windowed impulse responses using the frequency sampling method.

  Follows the approach in:
  https://ccrma.stanford.edu/~jos/sasp/Windowing_Desired_Impulse_Response.html

  Args:
    magnitudes: Frequency transfer curve. Float32 Tensor of shape [batch,
      n_frames, n_frequencies] or [batch, n_frequencies]. The frequencies of the
      last dimension are ordered as [0, f_nyqist / (n_frames -1), ...,
      f_nyquist], where f_nyquist is (sample_rate / 2). Automatically splits the
      audio into equally sized frames to match frames in magnitudes.
    window_size: Size of the window to apply in the time domain. If window_size
      is less than 1, it defaults to the impulse_response size.

  Returns:
    impulse_response: Time-domain FIR filter of shape
      [batch, frames, window_size] or [batch, window_size].

  Raises:
    ValueError: If window size is larger than fft size.
  """
  # Get the IR (zero-phase form).
  magnitudes = tf.complex(magnitudes, tf.zeros_like(magnitudes))
  impulse_response = tf.signal.irfft(magnitudes)

  # Window and put in causal form.
  impulse_response = apply_window_to_impulse_response(impulse_response,
                                                      window_size)

  return impulse_response


def sinc(x, threshold=1e-20):
  """Normalized zero phase version (peak at zero)."""
  x = tf_float32(x)
  x = tf.where(tf.abs(x) < threshold, threshold * tf.ones_like(x), x)
  x = np.pi * x
  return tf.sin(x) / x


def sinc_impulse_response(cutoff_frequency, window_size=512, sample_rate=None):
  """Get a sinc impulse response for a set of low-pass cutoff frequencies.

  Args:
    cutoff_frequency: Frequency cutoff for low-pass sinc filter. If the
      sample_rate is given, cutoff_frequency is in Hertz. If sample_rate is
      None, cutoff_frequency is normalized ratio (frequency/nyquist) in the
      range [0, 1.0]. Shape [batch_size, n_time, 1].
    window_size: Size of the Hamming window to apply to the impulse.
    sample_rate: Optionally provide the sample rate.

  Returns:
    impulse_response: A series of impulse responses. Shape
      [batch_size, n_time, (window_size // 2) * 2 + 1].
  """
  # Convert frequency to samples/sample_rate [0, Nyquist] -> [0, 1].
  if sample_rate is not None:
    cutoff_frequency *= 2.0 / float(sample_rate)

  # Create impulse response axis.
  half_size = window_size // 2
  full_size = half_size * 2 + 1
  idx = tf.range(-half_size, half_size + 1.0, dtype=tf.float32)
  idx = idx[tf.newaxis, tf.newaxis, :]

  # Compute impulse response.
  impulse_response = sinc(cutoff_frequency * idx)

  # Window the impulse response.
  window = tf.signal.hamming_window(full_size)
  window = tf.broadcast_to(window, impulse_response.shape)
  impulse_response = window * tf.math.real(impulse_response)

  # Normalize for unity gain.
  impulse_response /= tf.reduce_sum(impulse_response, axis=-1, keepdims=True)
  return impulse_response


def frequency_filter(audio: tf.Tensor,
                     magnitudes: tf.Tensor,
                     window_size: int = 0,
                     padding: Text = 'same') -> tf.Tensor:
  """Filter audio with a finite impulse response filter.

  Args:
    audio: Input audio. Tensor of shape [batch, audio_timesteps].
    magnitudes: Frequency transfer curve. Float32 Tensor of shape [batch,
      n_frames, n_frequencies] or [batch, n_frequencies]. The frequencies of the
      last dimension are ordered as [0, f_nyqist / (n_frames -1), ...,
      f_nyquist], where f_nyquist is (sample_rate / 2). Automatically splits the
      audio into equally sized frames to match frames in magnitudes.
    window_size: Size of the window to apply in the time domain. If window_size
      is less than 1, it is set as the default (n_frequencies).
    padding: Either 'valid' or 'same'. For 'same' the final output to be the
      same size as the input audio (audio_timesteps). For 'valid' the audio is
      extended to include the tail of the impulse response (audio_timesteps +
      window_size - 1).

  Returns:
    Filtered audio. Tensor of shape
        [batch, audio_timesteps + window_size - 1] ('valid' padding) or shape
        [batch, audio_timesteps] ('same' padding).
  """
  impulse_response = frequency_impulse_response(magnitudes,
                                                window_size=window_size)
  return fft_convolve(audio, impulse_response, padding=padding)


def sinc_filter(audio: tf.Tensor,
                cutoff_frequency: tf.Tensor,
                window_size: int = 512,
                sample_rate: int = None,
                padding: Text = 'same') -> tf.Tensor:
  """Filter audio with sinc low-pass filter.

  Args:
    audio: Input audio. Tensor of shape [batch, audio_timesteps].
    cutoff_frequency: Frequency cutoff for low-pass sinc filter. If the
      sample_rate is given, cutoff_frequency is in Hertz. If sample_rate is
      None, cutoff_frequency is normalized ratio (frequency/nyquist) in the
      range [0, 1.0]. Shape [batch_size, n_time, 1].
    window_size: Size of the Hamming window to apply to the impulse.
    sample_rate: Optionally provide the sample rate.
    padding: Either 'valid' or 'same'. For 'same' the final output to be the
      same size as the input audio (audio_timesteps). For 'valid' the audio is
      extended to include the tail of the impulse response (audio_timesteps +
      window_size - 1).

  Returns:
    Filtered audio. Tensor of shape
      [batch, audio_timesteps + window_size - 1] ('valid' padding) or shape
      [batch, audio_timesteps] ('same' padding).
  """
  impulse_response = sinc_impulse_response(cutoff_frequency,
                                           window_size=window_size,
                                           sample_rate=sample_rate)
  return fft_convolve(audio, impulse_response, padding=padding)
