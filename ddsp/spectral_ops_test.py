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
"""Tests for ddsp.losses."""

from absl.testing import parameterized
from ddsp import spectral_ops
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


class ComputeF0AndLoudnessTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    """Creates some common default values for the test sinusoid."""
    super().setUp()
    self.amp = 0.75
    self.frequency = 440.0
    self.frame_rate = 250

  def _gen_sinusoid(self, sample_rate, audio_len_sec):
    x = np.linspace(0, audio_len_sec, audio_len_sec * sample_rate)
    audio_sin = self.amp * (np.sin(2 * np.pi * self.frequency * x))
    return audio_sin

  @parameterized.named_parameters(
      ('16k_2.1secs', 16000, 2.1),
      ('24k_2.1secs', 24000, 2.1),
      ('44.1k_2.1secs', 44100, 2.1),
      ('48k_2.1secs', 48000, 2.1),
      ('16k_4secs', 16000, 4),
      ('24k_4secs', 24000, 4),
      ('44.1k_4secs', 44100, 4),
      ('48k_4secs', 48000, 4),
  )
  def test_compute_f0_at_sample_rate(self, sample_rate, audio_len_sec):
    audio_sin = self._gen_sinusoid(sample_rate, audio_len_sec)
    f0_hz, f0_confidence = spectral_ops.compute_f0(audio_sin, sample_rate,
                                                   self.frame_rate)
    expected_f0_hz_and_f0_conf_len = int(self.frame_rate * audio_len_sec)
    self.assertLen(f0_hz, expected_f0_hz_and_f0_conf_len)
    self.assertLen(f0_confidence, expected_f0_hz_and_f0_conf_len)

  @parameterized.named_parameters(
      ('16k_2.1secs', 16000, 2.1),
      ('24k_2.1secs', 24000, 2.1),
      ('48k_2.1secs', 48000, 2.1),
      ('16k_4secs', 16000, 4),
      ('24k_4secs', 24000, 4),
      ('48k_4secs', 48000, 4),
  )
  def test_compute_loudness_at_sample_rate(self, sample_rate, audio_len_sec):
    audio_sin = self._gen_sinusoid(sample_rate, audio_len_sec)
    loudness = spectral_ops.compute_loudness(audio_sin, sample_rate,
                                             self.frame_rate)
    expected_loudness_len = int(self.frame_rate * audio_len_sec)
    self.assertLen(loudness, expected_loudness_len)

  @parameterized.named_parameters(
      ('441.k_2.1secs', 44100, 2.1),
      ('441.k_4secs', 44100, 4),
  )
  def test_compute_loudness_at_indivisible_sample_rate(self, sample_rate,
                                                       audio_len_sec):
    audio_sin = self._gen_sinusoid(sample_rate, audio_len_sec)
    with self.assertRaises(ValueError):
      spectral_ops.compute_loudness(audio_sin, sample_rate, self.frame_rate)


if __name__ == '__main__':
  tf.test.main()
