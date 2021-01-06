# Copyright 2021 The DDSP Authors.
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
"""Tests for ddsp.losses."""

from absl.testing import parameterized
from ddsp import spectral_ops
import numpy as np
import tensorflow.compat.v2 as tf


def gen_np_sinusoid(frequency, amp, sample_rate, audio_len_sec):
  x = np.linspace(0, audio_len_sec, int(audio_len_sec * sample_rate))
  audio_sin = amp * (np.sin(2 * np.pi * frequency * x))
  return audio_sin


def gen_np_batched_sinusoids(frequency, amp, sample_rate, audio_len_sec,
                             batch_size):
  batch_sinusoids = [
      gen_np_sinusoid(frequency, amp, sample_rate, audio_len_sec)
      for _ in range(batch_size)
  ]
  return np.array(batch_sinusoids)


class STFTTest(tf.test.TestCase):

  def test_tf_and_np_are_consistent(self):
    amp = 1e-2
    audio = amp * (np.random.rand(64000).astype(np.float32) * 2.0 - 1.0)
    frame_size = 2048
    hop_size = 128
    overlap = 1.0 - float(hop_size) / frame_size
    pad_end = True

    s_np = spectral_ops.stft_np(
        audio, frame_size=frame_size, overlap=overlap, pad_end=pad_end)

    s_tf = spectral_ops.stft(
        audio, frame_size=frame_size, overlap=overlap, pad_end=pad_end)

    # TODO(jesseengel): The phase comes out a little different, figure out why.
    self.assertAllClose(np.abs(s_np), np.abs(s_tf), rtol=1e-3, atol=1e-3)


class DiffTest(tf.test.TestCase):

  def test_shape_is_correct(self):
    n_batch = 2
    n_time = 125
    n_freq = 100
    mag = tf.ones([n_batch, n_time, n_freq])

    diff = spectral_ops.diff
    delta_t = diff(mag, axis=1)
    self.assertEqual(delta_t.shape[1], mag.shape[1] - 1)
    delta_delta_t = diff(delta_t, axis=1)
    self.assertEqual(delta_delta_t.shape[1], mag.shape[1] - 2)
    delta_f = diff(mag, axis=2)
    self.assertEqual(delta_f.shape[2], mag.shape[2] - 1)
    delta_delta_f = diff(delta_f, axis=2)
    self.assertEqual(delta_delta_f.shape[2], mag.shape[2] - 2)


class LoudnessTest(tf.test.TestCase):

  def test_tf_and_np_are_consistent(self):
    amp = 1e-2
    audio = amp * (np.random.rand(64000).astype(np.float32) * 2.0 - 1.0)
    frame_size = 2048
    frame_rate = 250

    ld_tf = spectral_ops.compute_loudness(
        audio, n_fft=frame_size, frame_rate=frame_rate, use_tf=True)

    ld_np = spectral_ops.compute_loudness(
        audio, n_fft=frame_size, frame_rate=frame_rate, use_tf=False)

    self.assertAllClose(np.abs(ld_np), np.abs(ld_tf), rtol=1e-3, atol=1e-3)


class PadOrTrimVectorToExpectedLengthTest(parameterized.TestCase,
                                          tf.test.TestCase):

  @parameterized.named_parameters(
      ('np_1d', False, 1),
      ('np_2d', False, 2),
      ('tf_1d', True, 1),
      ('tf_2d', True, 2),
  )
  def test_pad_or_trim_vector_to_expected_length(self, use_tf, num_dims):
    vector_len = 10
    padded_vector_expected_len = 15
    trimmed_vector_expected_len = 4

    # Generate target vectors for testing
    vector = np.ones(vector_len) + np.random.uniform()
    num_pad = padded_vector_expected_len - vector_len
    target_padded = np.concatenate([vector, np.zeros(num_pad)])
    target_trimmed = vector[:trimmed_vector_expected_len]

    # Make a batch of target vectors
    if num_dims > 1:
      batch_size = 16
      vector = np.tile(vector, (batch_size, 1))
      target_padded = np.tile(target_padded, (batch_size, 1))
      target_trimmed = np.tile(target_trimmed, (batch_size, 1))

    vector_padded = spectral_ops.pad_or_trim_to_expected_length(
        vector, padded_vector_expected_len, use_tf=use_tf)
    vector_trimmmed = spectral_ops.pad_or_trim_to_expected_length(
        vector, trimmed_vector_expected_len, use_tf=use_tf)
    self.assertAllClose(target_padded, vector_padded)
    self.assertAllClose(target_trimmed, vector_trimmmed)


class ComputeFeaturesTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    """Creates some common default values for the test sinusoid."""
    super().setUp()
    self.amp = 0.75
    self.frequency = 440.0
    self.frame_rate = 250

  @parameterized.named_parameters(
      ('16k_.21secs', 16000, .21),
      ('24k_.21secs', 24000, .21),
      ('44.1k_.21secs', 44100, .21),
      ('48k_.21secs', 48000, .21),
      ('16k_.4secs', 16000, .4),
      ('24k_.4secs', 24000, .4),
      ('44.1k_.4secs', 44100, .4),
      ('48k_.4secs', 48000, .4),
  )
  def test_compute_f0_at_sample_rate(self, sample_rate, audio_len_sec):
    audio_sin = gen_np_sinusoid(self.frequency, self.amp, sample_rate,
                                audio_len_sec)
    f0_hz, f0_confidence = spectral_ops.compute_f0(audio_sin, sample_rate,
                                                   self.frame_rate)
    expected_f0_hz_and_f0_conf_len = int(self.frame_rate * audio_len_sec)
    self.assertLen(f0_hz, expected_f0_hz_and_f0_conf_len)
    self.assertLen(f0_confidence, expected_f0_hz_and_f0_conf_len)
    self.assertTrue(np.all(np.isfinite(f0_hz)))
    self.assertTrue(np.all(np.isfinite(f0_confidence)))

  @parameterized.named_parameters(
      ('16k_.21secs', 16000, .21),
      ('24k_.21secs', 24000, .21),
      ('48k_.21secs', 48000, .21),
      ('16k_.4secs', 16000, .4),
      ('24k_.4secs', 24000, .4),
      ('48k_.4secs', 48000, .4),
  )
  def test_compute_loudness_at_sample_rate_1d(self, sample_rate, audio_len_sec):
    audio_sin = gen_np_sinusoid(self.frequency, self.amp, sample_rate,
                                audio_len_sec)
    expected_loudness_len = int(self.frame_rate * audio_len_sec)

    for use_tf in [False, True]:
      loudness = spectral_ops.compute_loudness(
          audio_sin, sample_rate, self.frame_rate, use_tf=use_tf)
      self.assertLen(loudness, expected_loudness_len)
      self.assertTrue(np.all(np.isfinite(loudness)))

  @parameterized.named_parameters(
      ('16k_.21secs', 16000, .21),
      ('24k_.21secs', 24000, .21),
      ('48k_.21secs', 48000, .21),
      ('16k_.4secs', 16000, .4),
      ('24k_.4secs', 24000, .4),
      ('48k_.4secs', 48000, .4),
  )
  def test_compute_loudness_at_sample_rate_2d(self, sample_rate, audio_len_sec):
    batch_size = 8
    audio_sin_batch = gen_np_batched_sinusoids(self.frequency, self.amp,
                                               sample_rate, audio_len_sec,
                                               batch_size)
    expected_loudness_len = int(self.frame_rate * audio_len_sec)

    for use_tf in [False, True]:
      loudness_batch = spectral_ops.compute_loudness(
          audio_sin_batch, sample_rate, self.frame_rate, use_tf=use_tf)

      self.assertEqual(loudness_batch.shape[0], batch_size)
      self.assertEqual(loudness_batch.shape[1], expected_loudness_len)
      self.assertTrue(np.all(np.isfinite(loudness_batch)))

      # Check if batched loudness is equal to equivalent single computations
      audio_sin = gen_np_sinusoid(self.frequency, self.amp, sample_rate,
                                  audio_len_sec)
      loudness_target = spectral_ops.compute_loudness(
          audio_sin, sample_rate, self.frame_rate, use_tf=use_tf)
      loudness_batch_target = np.tile(loudness_target, (batch_size, 1))
      # Allow tolerance within 1dB
      self.assertAllClose(loudness_batch, loudness_batch_target, atol=1, rtol=1)

  @parameterized.named_parameters(
      ('16k_.21secs', 16000, .21),
      ('24k_.21secs', 24000, .21),
      ('48k_.21secs', 48000, .21),
      ('16k_.4secs', 16000, .4),
      ('24k_.4secs', 24000, .4),
      ('48k_.4secs', 48000, .4),
  )
  def test_tf_compute_loudness_at_sample_rate(self, sample_rate, audio_len_sec):
    audio_sin = gen_np_sinusoid(self.frequency, self.amp, sample_rate,
                                audio_len_sec)
    loudness = spectral_ops.compute_loudness(audio_sin, sample_rate,
                                             self.frame_rate)
    expected_loudness_len = int(self.frame_rate * audio_len_sec)
    self.assertLen(loudness, expected_loudness_len)
    self.assertTrue(np.all(np.isfinite(loudness)))

  @parameterized.named_parameters(
      ('44.1k_.21secs', 44100, .21),
      ('44.1k_.4secs', 44100, .4),
  )
  def test_compute_loudness_indivisible_rates_raises_error(
      self, sample_rate, audio_len_sec):
    audio_sin = gen_np_sinusoid(self.frequency, self.amp, sample_rate,
                                audio_len_sec)

    for use_tf in [False, True]:
      with self.assertRaises(ValueError):
        spectral_ops.compute_loudness(
            audio_sin, sample_rate, self.frame_rate, use_tf=use_tf)

  @parameterized.named_parameters(
      ('16k_.21secs', 16000, .21),
      ('24k_.21secs', 24000, .21),
      ('48k_.21secs', 48000, .21),
      ('16k_.4secs', 16000, .4),
      ('24k_.4secs', 24000, .4),
      ('48k_.4secs', 48000, .4),
  )
  def test_compute_rms_energy(self, sample_rate, audio_len_sec):
    audio_sin = gen_np_sinusoid(self.frequency, self.amp, sample_rate,
                                audio_len_sec)
    expected_rmse_len = int(self.frame_rate * audio_len_sec)

    rms_energy = spectral_ops.compute_rms_energy(
        audio_sin, sample_rate, self.frame_rate)
    self.assertLen(rms_energy, expected_rmse_len)
    self.assertTrue(np.all(np.isfinite(rms_energy)))

  @parameterized.named_parameters(
      ('16k_.21secs', 16000, .21),
      ('24k_.21secs', 24000, .21),
      ('48k_.21secs', 48000, .21),
      ('16k_.4secs', 16000, .4),
      ('24k_.4secs', 24000, .4),
      ('48k_.4secs', 48000, .4),
  )
  def test_compute_power(self, sample_rate, audio_len_sec):
    audio_sin = gen_np_sinusoid(self.frequency, self.amp, sample_rate,
                                audio_len_sec)
    expected_power_len = int(self.frame_rate * audio_len_sec)

    power = spectral_ops.compute_power(
        audio_sin, sample_rate, self.frame_rate)
    self.assertLen(power, expected_power_len)
    self.assertTrue(np.all(np.isfinite(power)))


if __name__ == '__main__':
  tf.test.main()
