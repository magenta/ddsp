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

"""Tests for ddsp.losses."""

from absl.testing import parameterized
from ddsp import spectral_ops
from ddsp.test_util import gen_np_sinusoid
import numpy as np
import tensorflow.compat.v2 as tf


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
    self.frame_size = 512

  def expected_f0_length(self, audio, padding):
    n_t = audio.shape[-1]
    frame_size = spectral_ops.CREPE_FRAME_SIZE
    hop_size = int(16000 // self.frame_rate)
    expected_len, _ = spectral_ops.get_framed_lengths(
        n_t, frame_size, hop_size, padding)
    return expected_len

  def expected_db_length(self, audio, sr, padding):
    n_t = audio.shape[-1]
    hop_size = int(sr // self.frame_rate)
    expected_len, _ = spectral_ops.get_framed_lengths(
        n_t, self.frame_size, hop_size, padding)
    return expected_len

  @parameterized.named_parameters(
      ('same_.21secs', 'same', .21),
      ('same_.4secs', 'same', .4),
      ('center_.21secs', 'center', .21),
      ('center_.4secs', 'center', .4),
      ('valid_.21secs', 'valid', .21),
      ('valid_.4secs', 'valid', .4),
  )
  def test_compute_f0(self, padding, audio_len_sec):
    """Ensure that compute_f0 (crepe) has expected output shape."""
    sr = 16000
    audio_sin = gen_np_sinusoid(self.frequency, self.amp, sr, audio_len_sec)
    expected_len = self.expected_f0_length(audio_sin, padding)
    f0_hz, f0_confidence = spectral_ops.compute_f0(
        audio_sin, self.frame_rate, viterbi=True, padding=padding)
    self.assertLen(f0_hz, expected_len)
    self.assertLen(f0_confidence, expected_len)
    self.assertTrue(np.all(np.isfinite(f0_hz)))
    self.assertTrue(np.all(np.isfinite(f0_confidence)))

  def test_batch_compute_db(self):
    """Ensure that compute_(loudness/power) can work on a batch."""
    batch_size = 2
    sample_rate = 16000
    audio_len_sec = 0.21
    padding = 'same'
    audio_sin = gen_np_sinusoid(self.frequency, self.amp, sample_rate,
                                audio_len_sec)
    expected_len = self.expected_db_length(audio_sin, sample_rate, padding)
    audio_batch = tf.tile(audio_sin[None, :], [batch_size, 1])
    loudness = spectral_ops.compute_loudness(
        audio_batch, sample_rate, self.frame_rate, self.frame_size,
        padding=padding)
    power = spectral_ops.compute_power(
        audio_batch, sample_rate, self.frame_rate, self.frame_size,
        padding=padding)
    self.assertLen(loudness.shape, 2)
    self.assertLen(power.shape, 2)
    self.assertEqual(batch_size, loudness.shape[0])
    self.assertEqual(batch_size, power.shape[0])
    self.assertEqual(expected_len, loudness.shape[1])
    self.assertEqual(expected_len, power.shape[1])

  def test_compute_loudness_tf_np(self):
    """Ensure that compute_loudness is the same output for np and tf."""
    sample_rate = 16000
    audio_len_sec = 0.21
    audio_sin = gen_np_sinusoid(self.frequency, self.amp, sample_rate,
                                audio_len_sec)
    loudness_tf = spectral_ops.compute_loudness(
        audio_sin, sample_rate, self.frame_rate, self.frame_size, use_tf=True)
    loudness_np = spectral_ops.compute_loudness(
        audio_sin, sample_rate, self.frame_rate, self.frame_size, use_tf=False)
    # Allow tolerance within 1dB
    self.assertAllClose(loudness_tf.numpy(), loudness_np, atol=1, rtol=1)

  @parameterized.named_parameters(
      ('16k_.21secs', 16000, .21),
      ('24k_.21secs', 24000, .21),
      ('44.1k_.21secs', 44100, .21),
      ('16k_.4secs', 16000, .4),
      ('24k_.4secs', 24000, .4),
      ('44.1k_.4secs', 44100, .4),
  )
  def test_compute_loudness(self, sample_rate, audio_len_sec):
    """Ensure that compute_loudness has expected output shape."""
    padding = 'center'
    audio_sin = gen_np_sinusoid(self.frequency, self.amp, sample_rate,
                                audio_len_sec)
    expected_len = self.expected_db_length(audio_sin, sample_rate, padding)
    loudness = spectral_ops.compute_loudness(
        audio_sin, sample_rate, self.frame_rate, self.frame_size,
        padding=padding)
    self.assertLen(loudness, expected_len)
    self.assertTrue(np.all(np.isfinite(loudness)))

  @parameterized.named_parameters(
      ('same', 'same'),
      ('valid', 'valid'),
      ('center', 'center'),
  )
  def test_compute_loudness_padding(self, padding):
    """Ensure that compute_loudness works with different paddings."""
    sample_rate = 16000
    audio_len_sec = 0.21
    audio_sin = gen_np_sinusoid(self.frequency, self.amp, sample_rate,
                                audio_len_sec)
    expected_len = self.expected_db_length(audio_sin, sample_rate, padding)
    loudness = spectral_ops.compute_loudness(
        audio_sin, sample_rate, self.frame_rate, self.frame_size,
        padding=padding)
    self.assertLen(loudness, expected_len)
    self.assertTrue(np.all(np.isfinite(loudness)))

  @parameterized.named_parameters(
      ('16k_.21secs', 16000, .21),
      ('24k_.21secs', 24000, .21),
      ('44.1k_.21secs', 44100, .21),
      ('16k_.4secs', 16000, .4),
      ('24k_.4secs', 24000, .4),
      ('44.1k_.4secs', 44100, .4),
  )
  def test_compute_rms_energy(self, sample_rate, audio_len_sec):
    """Ensure that compute_rms_energy has expected output shape."""
    padding = 'center'
    audio_sin = gen_np_sinusoid(self.frequency, self.amp, sample_rate,
                                audio_len_sec)
    expected_len = self.expected_db_length(audio_sin, sample_rate, padding)
    rms_energy = spectral_ops.compute_rms_energy(
        audio_sin, sample_rate, self.frame_rate, self.frame_size,
        padding=padding)
    self.assertLen(rms_energy, expected_len)
    self.assertTrue(np.all(np.isfinite(rms_energy)))

  @parameterized.named_parameters(
      ('same', 'same'),
      ('valid', 'valid'),
      ('center', 'center'),
  )
  def test_compute_power_padding(self, padding):
    """Ensure that compute_power (-> +rms) work with different paddings."""
    sample_rate = 16000
    audio_len_sec = 0.21
    audio_sin = gen_np_sinusoid(self.frequency, self.amp, sample_rate,
                                audio_len_sec)
    expected_len = self.expected_db_length(audio_sin, sample_rate, padding)
    power = spectral_ops.compute_power(
        audio_sin, sample_rate, self.frame_rate, self.frame_size,
        padding=padding)
    self.assertLen(power, expected_len)
    self.assertTrue(np.all(np.isfinite(power)))


class PadTest(parameterized.TestCase, tf.test.TestCase):

  def test_pad_end_stft_is_consistent(self):
    """Ensure that spectral_ops.pad('same') is same as stft(pad_end=True)."""
    frame_size = 200
    hop_size = 180
    audio = tf.random.normal([1, 1000])
    padded_audio = spectral_ops.pad(audio, frame_size, hop_size, 'same')
    s_pad_end = tf.signal.stft(audio, frame_size, hop_size, pad_end=True)
    s_same = tf.signal.stft(padded_audio, frame_size, hop_size, pad_end=False)
    self.assertAllClose(np.abs(s_pad_end), np.abs(s_same), rtol=1e-3, atol=1e-3)

  @parameterized.named_parameters(
      ('valid_odd', 'valid', 180),
      ('same_odd', 'same', 180),
      ('center_odd', 'center', 180),
      ('valid_even', 'valid', 200),
      ('same_even', 'same', 200),
      ('center_even', 'center', 200),
  )
  def test_padding_shapes_are_correct(self, padding, hop_size):
    """Ensure that pad() and get_framed_lengths() have correct shapes."""
    frame_size = 200
    n_t = 1000
    audio = tf.random.normal([1, n_t])
    padded_audio = spectral_ops.pad(audio, frame_size, hop_size, padding)
    n_t_pad = padded_audio.shape[1]

    frames = tf.signal.frame(padded_audio, frame_size, hop_size)
    n_frames = frames.shape[1]

    exp_n_frames, exp_n_t_pad = spectral_ops.get_framed_lengths(
        n_t, frame_size, hop_size, padding)

    self.assertEqual(n_frames, exp_n_frames)
    self.assertEqual(n_t_pad, exp_n_t_pad)


if __name__ == '__main__':
  tf.test.main()
