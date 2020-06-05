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
"""Tests for ddsp.training.eval_util."""

from absl.testing import parameterized
from ddsp.spectral_ops_test import gen_np_sinusoid
from ddsp.training.metrics import compute_audio_features
import numpy as np
import tensorflow.compat.v2 as tf


class ComputeAudioFeaturesTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    """Create some common default values for the test sinusoid."""
    super().setUp()
    self.amp = 0.75
    self.frequency = 440.0
    self.frame_rate = 250

  def validate_output_shapes(self, audio_features, expected_feature_lengths):
    for feat, expected_len in expected_feature_lengths.items():
      arr = audio_features[feat]
      try:
        self.assertLen(arr, expected_len)
      except AssertionError as e:
        raise AssertionError('%s feature: %s' % (e, feat))
      self.assertTrue(np.all(np.isfinite(arr)))

  @parameterized.named_parameters(
      ('16k_.21secs', 16000, .21),
      ('24k_.21secs', 24000, .21),
      ('48k_.21secs', 48000, .21),
      ('16k_.4secs', 16000, .4),
      ('24k_.4secs', 24000, .4),
      ('48k_.4secs', 48000, .4),
  )
  def test_correct_shape_compute_af_at_sample_rate(self, sample_rate,
                                                   audio_len_sec):
    audio_sin = gen_np_sinusoid(self.frequency, self.amp, sample_rate,
                                audio_len_sec)
    audio_features = compute_audio_features(
        audio_sin, sample_rate=sample_rate, frame_rate=self.frame_rate)

    expected_f0_and_loudness_length = int(audio_len_sec * self.frame_rate)
    self.validate_output_shapes(
        audio_features, {
            'audio': audio_len_sec * sample_rate,
            'f0_hz': expected_f0_and_loudness_length,
            'f0_confidence': expected_f0_and_loudness_length,
            'loudness_db': expected_f0_and_loudness_length,
        })

  @parameterized.named_parameters(
      ('44.1k_.21secs', 44100, .21),
      ('44.1k_.4secs', 44100, .4),
  )
  def test_indivisible_rates_raises_error_compute_af(self, sample_rate,
                                                     audio_len_sec):
    audio_sin = gen_np_sinusoid(self.frequency, self.amp, sample_rate,
                                audio_len_sec)

    with self.assertRaises(ValueError):
      compute_audio_features(
          audio_sin, sample_rate=sample_rate, frame_rate=self.frame_rate)


if __name__ == '__main__':
  tf.test.main()
