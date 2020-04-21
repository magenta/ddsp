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
"""Tests for ddsp.core."""

from absl.testing import parameterized
from ddsp import core
import librosa
import numpy as np
from scipy import signal
import tensorflow.compat.v2 as tf


class UtilitiesTest(parameterized.TestCase, tf.test.TestCase):

  def test_midi_to_hz_is_accurate(self):
    """Tests converting between MIDI values and their frequencies in hertz."""
    midi = np.arange(128)
    librosa_hz = librosa.midi_to_hz(midi)
    tf_hz = core.midi_to_hz(midi)
    self.assertAllClose(librosa_hz, tf_hz)

  def test_hz_to_midi_is_accurate(self):
    """Tests converting between MIDI values and their frequencies in hertz."""
    hz = np.linspace(20.0, 20000.0, 128)
    librosa_midi = librosa.hz_to_midi(hz)
    tf_midi = core.hz_to_midi(hz)
    self.assertAllClose(librosa_midi, tf_midi)

  @parameterized.named_parameters(
      ('clip', True), ('no_clip', False)
  )
  def test_midi_to_unit_is_accurate(self, clip):
    """Tests converting between MIDI values and the unit interval.

    Args:
      clip: Whether to clip the output to [0.0, 1.0].
    """
    midi_min, midi_max = 20.0, 90.0
    midi = np.linspace(0.0, 127.0, 1000)
    np_unit = (midi - midi_min) / (midi_max - midi_min)
    np_unit = np.clip(np_unit, 0.0, 1.0) if clip else np_unit
    tf_unit = core.midi_to_unit(
        midi, midi_min=midi_min, midi_max=midi_max, clip=clip)
    self.assertAllClose(tf_unit, np_unit)

  @parameterized.named_parameters(
      ('clip', True), ('no_clip', False)
  )
  def test_unit_to_midi_is_accurate(self, clip):
    """Tests converting between the unit interval and MIDI values.

    Args:
      clip: Whether to clip the input to [0.0, 1.0].
    """
    midi_min, midi_max = 20.0, 90.0
    unit = np.linspace(-1.0, 2.0, 1000)
    np_midi = np.clip(unit, 0.0, 1.0) if clip else unit
    np_midi = midi_min + (midi_max - midi_min) * np_midi
    tf_midi = core.unit_to_midi(
        unit, midi_min=midi_min, midi_max=midi_max, clip=clip)
    self.assertAllClose(tf_midi, np_midi)

  def test_unit_to_hz_is_accurate(self):
    """Tests converting between unit interval and their frequencies in hertz."""
    hz_min = 20.0
    hz_max = 1000.0
    unit = np.linspace(0.0, 1.0, 128)
    np_hz = np.logspace(np.log10(hz_min), np.log10(hz_max), 128)
    tf_hz = core.unit_to_hz(unit, hz_min, hz_max)
    self.assertAllClose(np_hz, tf_hz)

  def test_hz_to_unit_is_accurate(self):
    """Tests converting between frequencies in hertz and unit interval."""
    hz_min = 20.0
    hz_max = 1000.0
    hz = np.logspace(np.log10(hz_min), np.log10(hz_max), 128)
    np_unit = np.linspace(0.0, 1.0, 128)
    tf_unit = core.hz_to_unit(hz, hz_min, hz_max)
    self.assertAllClose(np_unit, tf_unit)


class ResampleTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    """Creates some common default values for the tests."""
    super().setUp()
    self.n_smaller = 5
    self.n_larger = 16000

  @parameterized.named_parameters(
      ('1-D', 1), ('2-D', 2), ('3-D', 3), ('4-D', 4),
  )
  def test_multi_dimensional_inputs(self, dimensions):
    """Test the shapes are correct for different dimensional inputs.

    Args:
      dimensions: The number of dimensions of the input test signal.
    """
    # Create test signal.
    inputs_shape = [self.n_smaller] * dimensions
    inputs = np.ones(inputs_shape)

    # Run through the resampling op.
    outputs = core.resample(inputs, self.n_larger)

    # Compute output shape.
    outputs_shape = inputs_shape
    if dimensions == 1:
      outputs_shape[0] = self.n_larger
    else:
      outputs_shape[1] = self.n_larger

    self.assertListEqual(list(outputs.shape), outputs_shape)

  @parameterized.named_parameters(
      ('1-D', 1), ('2-D', 2), ('3-D', 3), ('4-D', 4),
  )
  def test_window_only_allows_3d_inputs(self, dimensions):
    """Test that upsample_with_windows() disallows inputs that are not 3-D.

    Args:
      dimensions: The number of dimensions of the input test signal.
    """
    # Create test signal.
    inputs_shape = [self.n_smaller] * dimensions
    inputs = np.ones(inputs_shape)

    # Run through the resampling op.
    if dimensions != 3:
      with self.assertRaises(ValueError):
        outputs = core.upsample_with_windows(inputs, self.n_larger)
    else:
      outputs = core.upsample_with_windows(inputs, self.n_larger)
      outputs_shape = [self.n_smaller, self.n_larger, self.n_smaller]
      self.assertListEqual(list(outputs.shape), outputs_shape)

  def create_resampled_signals(self, n_before, n_after, add_endpoint, method):
    """Helper function to resample a test signal using core.resample().

    Args:
      n_before: Number of timesteps before resampling.
      n_after: Number of timesteps after resampling.
      add_endpoint: Add extra timestep at end of resampling.
      method: Method of resampling.

    Returns:
      before: Numpy array before resampling. Shape (n_before,).
      after: Numpy array after resampling. Shape (n_after,).
    """
    before = 1.0 - np.sin(np.linspace(0, np.pi, n_before))
    before = before[np.newaxis, :, np.newaxis]
    after = core.resample(
        before, n_after, method=method, add_endpoint=add_endpoint).numpy()
    return before[0, :, 0], after[0, :, 0]

  def assert_subsampled_close(self,
                              smaller,
                              larger,
                              add_endpoint,
                              threshold=1e-3):
    """Check subsampled high-resolution signal close to low-resolution signal.

    Args:
      smaller: Low resolution numpy array. Shape (size,).
      larger: High-resolution numpy array. Shape (size,).
      add_endpoint: Extra timestep has been added at end of resampling.
      threshold: Assertion threshold for all_close.
    """
    n_smaller = smaller.size
    n_larger = larger.size
    if add_endpoint:
      n_total = int(n_larger / n_smaller * (n_smaller - 1))
    else:
      n_total = n_larger - 1
    subsample_index = np.linspace(0, n_total, n_smaller).astype(int)
    larger_subsampled = larger[subsample_index]
    self.assertAllClose(larger_subsampled, smaller, atol=threshold)

  @parameterized.named_parameters(
      ('endpoint_linear', True, 'linear'),
      ('no_endpoint_linear', False, 'linear'),
      ('endpoint_cubic', True, 'cubic'),
      ('no_endpoint_cubic', False, 'cubic'),
      ('endpoint_window', True, 'window'),
      ('no_endpoint_window', False, 'window'),
  )
  def test_upsample_accuracy(self, add_endpoint, method):
    """Test that upsampling is accurate for different methods.

    Generates a sample signal and resamples it to a higher resolution.
    Compares the upsampled signal and original signal at corresponding
    subsampled points.

    Args:
      add_endpoint: Add extra timestep at end of resampling.
      method: Method of resampling.
    """
    before, after = self.create_resampled_signals(
        n_before=self.n_smaller,
        n_after=self.n_larger,
        add_endpoint=add_endpoint,
        method=method)
    self.assert_subsampled_close(
        smaller=before, larger=after, add_endpoint=add_endpoint)

  @parameterized.named_parameters(
      ('endpoint_linear', True, 'linear'),
      ('no_endpoint_linear', False, 'linear'),
      ('endpoint_cubic', True, 'cubic'),
      ('no_endpoint_cubic', False, 'cubic'),
  )
  def test_downsample_accuracy(self, add_endpoint, method):
    """Test that downsampling is accurate for different methods.

    Generates a signal and downsamples it to different resolution.
    Compares the downsampled signal and original signal at corresponding
    subsampled points. Don't test for `window` method because downsampling is
    not allowed.

    Args:
      add_endpoint: Add extra timestep at end of resampling.
      method: Method of resampling.
    """
    before, after = self.create_resampled_signals(
        n_before=self.n_larger,
        n_after=self.n_smaller,
        add_endpoint=add_endpoint,
        method=method)
    self.assert_subsampled_close(
        smaller=after, larger=before, add_endpoint=add_endpoint)

  @parameterized.named_parameters(
      ('endpoint', True),
      ('no_endpoint', False),
  )
  def test_window_checks_for_downsampling(self, add_endpoint):
    """Test that upsample_with_window raises ValueError for downsampling.

    Args:
      add_endpoint: Add extra timestep at end of resampling.
    """
    with self.assertRaises(ValueError):
      _ = self.create_resampled_signals(
          n_before=self.n_larger,
          n_after=self.n_smaller,
          add_endpoint=add_endpoint,
          method='window')

  @parameterized.named_parameters(
      ('endpoint', 5, True),
      ('no_endpoint', 6, False),
  )
  def test_window_allows_integer_upsampling_ratios(self, n_before,
                                                   add_endpoint):
    """Test that upsample_with_window runs for integer upsampling ratios.

    If add_endpoint is False, n_after must be divisible by n_before - 1 instead
    of n_before.

    Args:
      n_before: Number of points before resampling.
      add_endpoint: Add extra timestep at end of resampling.
    """
    _, after = self.create_resampled_signals(
        n_before=n_before,
        n_after=self.n_larger,
        add_endpoint=add_endpoint,
        method='window')
    self.assertEqual(self.n_larger, after.size)

  @parameterized.named_parameters(
      ('endpoint', 6, True),
      ('no_endpoint', 7, False),
  )
  def test_window_disallows_noninteger_upsampling_ratios(
      self, n_before, add_endpoint):
    """Test that upsample_with_window raises ValueError for non-integer ratios.

    If add_endpoint is False, n_after must be divisible by n_before - 1 instead
    of n_before.

    Args:
      n_before: Number of points before resampling.
      add_endpoint: Add extra timestep at end of resampling.
    """
    with self.assertRaises(ValueError):
      _ = self.create_resampled_signals(
          n_before=n_before,
          n_after=self.n_larger,
          add_endpoint=add_endpoint,
          method='window')

  @parameterized.named_parameters(
      ('linear', 'linear'),
      ('cubic', 'cubic'),
      ('window', 'window'),
  )
  def test_resample_allows_valid_method_arguments(self, method):
    """Tests resample runs with correct method names."""
    _, after = self.create_resampled_signals(
        n_before=self.n_smaller,
        n_after=self.n_larger,
        add_endpoint=True,
        method=method)
    self.assertEqual(self.n_larger, after.size)

  @parameterized.named_parameters(
      ('no_name', ''),
      ('bad_name', 'wiiinnndooww'),
  )
  def test_resample_disallows_invalid_method_arguments(self, method):
    """Tests resample() raises error for wrong method name."""
    with self.assertRaises(ValueError):
      _ = self.create_resampled_signals(
          n_before=self.n_smaller,
          n_after=self.n_larger,
          add_endpoint=True,
          method=method)


def create_wave_np(batch_size, frequencies, amplitudes, seconds, n_samples):
  """Helper function that synthesizes ground truth harmonic waves with numpy.

  Args:
    batch_size: Number of waves in the batch.
    frequencies: Array of harmonic frequencies in each wave. Shape (n_batch,
      n_time, n_harmonics). Units in Hertz.
    amplitudes: Array of amplitudes for each harmonic. Shape (n_batch, n_time,
      n_harmonics). Units in range 0 to 1.
    seconds: Length of the waves, in seconds.
    n_samples: Length of the waves, in samples.

  Returns:
    wave_np: An array of the synthesized waves. Shape (n_batch, n_samples).
  """
  wave_np = np.zeros([batch_size, n_samples])
  time = np.linspace(0, seconds, n_samples)
  n_harmonics = int(frequencies.shape[-1])
  for i in range(batch_size):
    for j in range(n_harmonics):
      rads_per_cycle = 2.0 * np.pi
      rads_per_sec = rads_per_cycle * frequencies[i, :, j]
      phase = time * rads_per_sec
      wave_np[i, :] += amplitudes[i, :, j] * np.sin(phase)
  return wave_np


class AdditiveSynthTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    """Creates some common default values for the tests."""
    super().setUp()
    self.batch_size = 2
    self.sample_rate = 16000
    self.seconds = 1.0
    self.n_samples = int(self.seconds) * self.sample_rate

  @parameterized.named_parameters(
      ('low_frequency', 2, 62.4, 5, 16000, 2),
      ('large_batch_size', 16, 100, 1, 8000, 0.5),
      ('high_frequency', 1, 2000, 2, 4000, 1.3),
  )
  def test_oscillator_bank_is_accurate(self, batch_size, fundamental_frequency,
                                       n_harmonics, sample_rate, seconds):
    """Test waveforms generated from oscillator_bank.

    Generates harmonic waveforms with tensorflow and numpy and tests that they
    are the same. Test over a range of inputs provided by the parameterized
    inputs.

    Args:
      batch_size: Size of the batch to synthesize.
      fundamental_frequency: Base frequency of the oscillator in Hertz.
      n_harmonics: Number of harmonics to synthesize.
      sample_rate: Sample rate of synthesis in samples per a second.
      seconds: Length of the generated test sample in seconds.
    """
    n_samples = int(sample_rate * seconds)
    seconds = float(n_samples) / sample_rate
    frequencies = fundamental_frequency * np.arange(1, n_harmonics + 1)
    amplitudes = 1.0 / n_harmonics * np.ones_like(frequencies)

    # Create tensors of frequencies and amplitudes for tf function.
    ones = np.ones([batch_size, n_samples, n_harmonics])
    frequency_envelopes = ones * frequencies[np.newaxis, np.newaxis, :]
    amplitude_envelopes = ones * amplitudes[np.newaxis, np.newaxis, :]

    # Create np test signal.
    wav_np = create_wave_np(batch_size, frequency_envelopes,
                            amplitude_envelopes, seconds, n_samples)

    wav_tf = core.oscillator_bank(
        frequency_envelopes, amplitude_envelopes, sample_rate=sample_rate)
    pad = 10  # Ignore edge effects.
    self.assertAllClose(wav_np[pad:-pad], wav_tf[pad:-pad])

  @parameterized.named_parameters(
      ('sum_sinusoids', True),
      ('no_sum_sinusoids', False),
  )
  def test_oscillator_bank_shape_is_correct(self, sum_sinusoids):
    """Tests that sum_sinusoids reduces the last dimension."""
    frequencies = np.array([1.0, 1.5, 2.0]) * 400.0
    amplitudes = np.ones_like(frequencies)

    # Create tensors of frequencies and amplitudes for tf function.
    ones = np.ones([self.batch_size, self.n_samples, 3])
    frequency_envelopes = ones * frequencies[np.newaxis, np.newaxis, :]
    amplitude_envelopes = ones * amplitudes[np.newaxis, np.newaxis, :]

    wav_tf = core.oscillator_bank(frequency_envelopes,
                                  amplitude_envelopes,
                                  sample_rate=self.sample_rate,
                                  sum_sinusoids=sum_sinusoids)
    if sum_sinusoids:
      expected_shape = [self.batch_size, self.n_samples]
    else:
      expected_shape = [self.batch_size, self.n_samples, 3]
    self.assertAllEqual(expected_shape, list(wav_tf.shape))

  @parameterized.named_parameters(
      ('low_sample_rate', 4000),
      ('16khz', 16000),
      ('cd_quality', 44100),
  )
  def test_silent_above_nyquist(self, sample_rate):
    """Tests that no freqencies above nyquist (sample_rate/2) are created."""
    nyquist = sample_rate / 2
    frequencies = np.array([1.1, 1.5, 2.0]) * nyquist
    amplitudes = np.ones_like(frequencies)

    # Create tensors of frequencies and amplitudes for tf function.
    ones = np.ones([self.batch_size, self.n_samples, 3])
    frequency_envelopes = ones * frequencies[np.newaxis, np.newaxis, :]
    amplitude_envelopes = ones * amplitudes[np.newaxis, np.newaxis, :]

    wav_tf = core.oscillator_bank(
        frequency_envelopes, amplitude_envelopes, sample_rate=sample_rate)
    wav_np = np.zeros_like(wav_tf)
    self.assertAllClose(wav_np, wav_tf)

  @parameterized.named_parameters(
      ('low_frequency', 2, 20, 0.1, 100),
      ('many_frames', 1, 100, 0.2, 1000),
      ('high_frequency', 4, 2000, 0.5, 100),
  )
  def test_harmonic_synthesis_is_accurate_one_frequency(self, batch_size,
                                                        fundamental_frequency,
                                                        amplitude, n_frames):
    """Tests generating a single sine wave with different frame parameters.

    Generates sine waveforms with tensorflow and numpy and tests that they are
    the same. Test over a range of inputs provided by the parameterized
    inputs.

    Args:
      batch_size: Size of the batch to synthesize.
      fundamental_frequency: Base frequency of the oscillator in Hertz.
      amplitude: Amplitude of each harmonic in the waveform.
      n_frames: Number of amplitude envelope frames.
    """
    frequencies = fundamental_frequency * np.ones([batch_size, n_frames, 1])
    amplitudes = amplitude * np.ones([batch_size, n_frames, 1])

    frequencies_np = fundamental_frequency * np.ones(
        [batch_size, self.n_samples, 1])
    amplitudes_np = amplitude * np.ones([batch_size, self.n_samples, 1])

    # Create np test signal.
    wav_np = create_wave_np(batch_size, frequencies_np, amplitudes_np,
                            self.seconds, self.n_samples)

    wav_tf = core.harmonic_synthesis(
        frequencies,
        amplitudes,
        n_samples=self.n_samples,
        sample_rate=self.sample_rate)
    pad = self.n_samples // n_frames  # Ignore edge effects.
    self.assertAllClose(wav_np[pad:-pad], wav_tf[pad:-pad])

  @parameterized.named_parameters(
      ('one_harmonic', 1),
      ('twenty_harmonics', 20),
      ('forty_harmonics', 40),
  )
  def test_harmonic_synthesis_is_accurate_multiple_harmonics(self, n_harmonics):
    """Tests generating a harmonic waveform with varying number of harmonics.

    Generates waveforms with tensorflow and numpy and tests that they are
    the same. Test over a range of inputs provided by the parameterized
    inputs.

    Args:
      n_harmonics: Number of harmonics to synthesize.
    """
    fundamental_frequency = 440.0
    amp = 0.1
    n_frames = 100

    harmonic_shifts = np.abs(np.random.randn(1, 1, n_harmonics))
    harmonic_distribution = np.abs(np.random.randn(1, 1, n_harmonics))

    frequencies_tf = fundamental_frequency * np.ones(
        [self.batch_size, n_frames, 1])
    amplitudes_tf = amp * np.ones([self.batch_size, n_frames, 1])
    harmonic_shifts_tf = np.tile(harmonic_shifts, [1, n_frames, 1])
    harmonic_distribution_tf = np.tile(harmonic_distribution, [1, n_frames, 1])

    # Create np test signal.
    frequencies_np = fundamental_frequency * np.ones(
        [self.batch_size, self.n_samples, 1])
    amplitudes_np = amp * np.ones([self.batch_size, self.n_samples, 1])
    frequencies_np = frequencies_np * harmonic_shifts
    amplitudes_np = amplitudes_np * harmonic_distribution
    wav_np = create_wave_np(self.batch_size, frequencies_np, amplitudes_np,
                            self.seconds, self.n_samples)

    wav_tf = core.harmonic_synthesis(
        frequencies_tf,
        amplitudes_tf,
        harmonic_shifts_tf,
        harmonic_distribution_tf,
        n_samples=self.n_samples,
        sample_rate=self.sample_rate)
    pad = self.n_samples // n_frames  # Ignore edge effects.
    self.assertAllClose(wav_np[pad:-pad], wav_tf[pad:-pad])


class InterpolatingLookupTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('high_frequency_wave', 1, 2048, 0, 10000, 1000),
      ('low_frequency_wave', 2, 1024, 0, 20000, 10),
      ('one_frame', 1, 2048, 1, 10000, 1000),
      ('many_frames', 1, 2048, 10000, 10000, 1000),
  )
  def test_linear_lookup_is_accurate(self, batch_size, n_wavetable, n_frames,
                                     n_samples, n_cycles):
    """Tests accuracy of linear interpolation lookup.

    Generate a sine wave from linear table lookup and compare to the analytic
    form. Error will vary with the size of the lookup table, but should stay
    below a threshold for moderate sized tables.

    Args:
      batch_size: Number of batch examples in the test phase signal.
      n_wavetable: Number of samples in the wavetable.
      n_frames: Number of frames in the wavetable.
      n_samples: Number of samples in the test phase signal.
      n_cycles: Number of cycles in the test phase signal.
    """
    threshold = 2e-3
    two_pi = 2.0 * np.pi
    wavetable = np.sin(np.linspace(0, two_pi, n_wavetable).astype(np.float32))
    wavetable = np.tile(wavetable[np.newaxis, :], [batch_size, 1])
    if n_frames > 0:
      wavetable = np.tile(wavetable[:, np.newaxis, :], [1, n_frames, 1])

    phase = np.linspace(0, n_cycles, n_samples).astype(np.float32) % 1.0
    phase = np.tile(phase[np.newaxis, :, np.newaxis], [batch_size, 1, 1])
    wav_np = np.sin(two_pi * phase)[:, :, 0]

    wav_tf = core.linear_lookup(phase, wavetable)

    difference = np.abs(wav_np - wav_tf).mean()
    self.assertLessEqual(difference, threshold)

  @parameterized.named_parameters(
      ('single_wavetable_no_frames', 1, 440.0, 0.5, 2048, 0),
      ('one_frame', 2, 1000.0, 0.1, 1024, 1),
      ('many_frames', 2, 1000.0, 0.1, 1024, 200),
  )
  def test_wavetable_synth_is_accurate(self, batch_size, frequency, amplitude,
                                       n_wavetable, wavetable_frames):
    """Tests accuracy of wavetable synthesizer.

    Generate a sine wave wavetable synthesizer and compare to the analytic
    form. Error will vary with the size of the lookup table, but should stay
    below a threshold for moderate sized tables.

    Args:
      batch_size: Number of batch examples in the test phase signal.
      frequency: Frequency of the carrier signal in Hertz.
      amplitude: Amplitude of the carrier signal.
      n_wavetable: Number of samples in the wavetable.
      wavetable_frames: Number of wavetables over time.
    """
    sample_rate = 16000
    seconds = 0.1
    n_samples = int(sample_rate * seconds)
    n_cycles = seconds * frequency
    n_frames = 100

    two_pi = 2.0 * np.pi
    wavetable = np.sin(np.linspace(0, two_pi, n_wavetable).astype(np.float32))
    wavetable = np.tile(wavetable[np.newaxis, :], [batch_size, 1])
    if wavetable_frames > 0:
      wavetable = np.tile(wavetable[:, np.newaxis, :], [1, wavetable_frames, 1])

    wav_np = amplitude * np.sin(two_pi * np.linspace(0, n_cycles, n_samples))
    wav_np = np.tile(wav_np[np.newaxis, :], [batch_size, 1]).astype(np.float32)

    amplitudes = np.ones([batch_size, n_frames, 1]) * amplitude
    frequencies = np.ones([batch_size, n_frames, 1]) * frequency

    wav_tf = core.wavetable_synthesis(frequencies, amplitudes, wavetable,
                                      n_samples, sample_rate)

    pad = n_samples // n_frames  # Ignore edge effects.
    difference = np.abs(wav_np[:, pad:-pad] - wav_tf[:, pad:-pad]).mean()
    threshold = 3e-2
    self.assertLessEqual(difference, threshold)

  @parameterized.named_parameters(
      ('short_delay', 1, 16000, 10),
      ('long_delay', 2, 4000, 1000),
  )
  def test_variable_length_delay_is_accurate(self, batch_size, n_samples,
                                             max_length):
    """Tests accuracy of variable length delay.

    Generate a sine wave and delay at various amounts. If max_length is equal
    to the period of oscillation, a half wave delay is equal to negation and
    full wave delay is equal to identity.

    Args:
      batch_size: Number of batch examples in the test phase signal.
      n_samples: Number of samples in the test signal.
      max_length: Maximimum delay in samples.
    """
    threshold = 1e-2
    # Start with a sin wave of same period as max_length.
    n_cycles = float(n_samples) / max_length
    wav_np = np.sin(np.linspace(0, 2.0 * np.pi * n_cycles, n_samples))
    wav_np = np.tile(wav_np[np.newaxis, :], [batch_size, 1]).astype(np.float32)

    # Three different decay amounts (none, half-wave, full-wave).
    ones = tf.ones_like(wav_np)[..., tf.newaxis]
    phase_no_delay = 0.0 * ones
    phase_half_delay = 0.5 * ones
    phase_full_delay = 1.0 * ones

    wav_tf_no_delay = core.variable_length_delay(phase_no_delay, wav_np,
                                                 max_length)
    wav_tf_half_delay = core.variable_length_delay(phase_half_delay, wav_np,
                                                   max_length)
    wav_tf_full_delay = core.variable_length_delay(phase_full_delay, wav_np,
                                                   max_length)

    for target, source in [(wav_np, wav_tf_no_delay),
                           (-wav_np, wav_tf_half_delay),
                           (wav_np, wav_tf_full_delay)]:
      # Ignore front of sample because of zero padding.
      difference = target[:, max_length:] - source[:, max_length:]
      difference = np.abs(difference).mean()
      self.assertLessEqual(difference, threshold)


class FiniteImpulseResponseTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    """Creates some common default values for the tests."""
    super().setUp()
    self.audio_size = 1000
    self.audio = np.random.randn(1, self.audio_size).astype(np.float32)

  @parameterized.named_parameters(
      ('ir_less_than_audio', 1000, 10),
      ('audio_less_than_ir', 10, 100),
  )
  def test_fft_convolve_is_accurate(self, audio_size, impulse_response_size):
    """Tests convolving signals using fast fourier transform (fft).

    Generate random signals and convolve using fft. Compare outputs to the
    implementation in scipy.signal.

    Args:
      audio_size: Size of the audio to convolve.
      impulse_response_size: Size of the impulse response to convolve.
    """

    # Create random signals to convolve.
    audio = np.ones([1, audio_size]).astype(np.float32)
    impulse_response = np.ones([1, impulse_response_size]).astype(np.float32)

    output_tf = core.fft_convolve(
        audio, impulse_response, padding='valid', delay_compensation=0)[0]

    output_np = signal.fftconvolve(audio[0], impulse_response[0])

    difference = output_np - output_tf
    total_difference = np.abs(difference).mean()
    threshold = 1e-3
    self.assertLessEqual(total_difference, threshold)

  @parameterized.named_parameters(
      ('unity_gain', 1.0),
      ('reduced_gain', 0.1),
  )
  def test_delay_compensation_corrects_group_delay(self, gain):
    """Test automatically compensating for group delay of linear phase filter.

    Genearate filters to shift entire signal by a constant gain. Test that
    filtered signal is in phase with original signal.

    Args:
      gain: Amount to scale the input signal.
    """
    # Create random signal to filter.
    output_np = gain * self.audio[0]
    n_frequencies = 1025
    window_size = 257

    magnitudes = gain * tf.ones([1, n_frequencies], dtype=tf.float32)
    impulse_response = core.frequency_impulse_response(magnitudes, window_size)
    output_tf = core.fft_convolve(
        self.audio, impulse_response, padding='same')[0]

    difference = output_np - output_tf
    total_difference = np.abs(difference).mean()
    threshold = 1e-3
    self.assertLessEqual(total_difference, threshold)

  def test_fft_convolve_checks_batch_size(self):
    """Tests fft_convolve() raises error for mismatched batch sizes."""
    # Create random signals to convolve with different batch sizes.
    impulse_response = tf.concat([self.audio, self.audio], axis=0)

    with self.assertRaises(ValueError):
      _ = core.fft_convolve(self.audio, impulse_response)

  @parameterized.named_parameters(
      ('same', 'same'),
      ('valid', 'valid'),
  )
  def test_fft_convolve_allows_valid_padding_arguments(self, padding):
    """Tests fft_convolve() runs for valid padding names."""
    result = core.fft_convolve(self.audio, self.audio, padding=padding)
    self.assertEqual(result.shape[0], 1)

  @parameterized.named_parameters(
      ('no_name', ''),
      ('bad_name', 'saaammmeee'),
  )
  def test_fft_convolve_disallows_invalid_padding_arguments(self, padding):
    """Tests fft_convolve() raises error for wrong padding name."""
    with self.assertRaises(ValueError):
      _ = core.fft_convolve(self.audio, self.audio, padding=padding)

  @parameterized.named_parameters(
      ('more_frames_than_timesteps', 1010),
      ('not_even_multiple', 999),
  )
  def test_fft_convolve_checks_number_of_frames(self, n_frames):
    """Tests fft_convolve() raises error for invalid number of framess."""
    # Create random signals to convolve with same batch sizes.
    impulse_response = tf.random.normal([1, n_frames, self.audio_size],
                                        dtype=tf.float32)
    with self.assertRaises(ValueError):
      _ = core.fft_convolve(self.audio, impulse_response)

  @parameterized.named_parameters(
      ('no_window', 2048, 0),
      ('typical_window', 2048, 257),
      ('atypical_window', 1024, 22),
      ('window_bigger', 1024, 2048),
  )
  def test_frequency_impulse_response_gives_correct_size(
      self, fft_size, window_size):
    """Tests generating impulse responses from a list of magnitudes.

    The output size should be determined by the window size, or fft_size if
    window size < 1.

    Args:
      fft_size: Size of the fft that generated the magnitudes.
      window_size: Size of window to apply to inverse fft.
    """
    # Create random signals to convolve.
    n_frequencies = fft_size // 2 + 1
    magnitudes = np.random.uniform(size=(1, n_frequencies)).astype(np.float32)

    impulse_response = core.frequency_impulse_response(magnitudes, window_size)

    target_size = fft_size
    if target_size > window_size >= 1:
      target_size = window_size
      is_even = target_size % 2 == 0
      target_size -= int(is_even)

    impulse_response_size = int(impulse_response.shape[-1])
    self.assertEqual(impulse_response_size, target_size)

  @parameterized.named_parameters(
      ('no_frames_no_window', 1025, 0, 0),
      ('no_frames_window', 1025, 0, 257),
      ('single_frame', 513, 1, 257),
      ('non_divisible_frames', 513, 13, 257),
      ('max_frames', 513, 1000, 257),
  )
  def test_frequency_filter_gives_correct_size(self, n_frequencies, n_frames,
                                               window_size):
    """Tests filtering signals with frequency sampling method.

    Generate random signals and magnitudes and filter using fft_convolve().

    Args:
      n_frequencies: Number of magnitudes.
      n_frames: Number of frames for a time-varying filter.
      window_size: Size of window for generating impulse responses.
    """
    # Create transfer function.
    if n_frames > 0:
      magnitudes = np.random.uniform(size=(1, n_frames,
                                           n_frequencies)).astype(np.float32)
    else:
      magnitudes = np.random.uniform(size=(1, n_frequencies)).astype(np.float32)

    audio_out = core.frequency_filter(
        self.audio, magnitudes, window_size=window_size, padding='same')

    audio_out_size = int(audio_out.shape[-1])
    self.assertEqual(audio_out_size, self.audio_size)


if __name__ == '__main__':
  tf.test.main()
