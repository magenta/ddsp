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
from ddsp.training.eval_util import compute_audio_features
import numpy as np
import tensorflow.compat.v2 as tf


class ComputeAudioFeaturesTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    """Creates some common default values for the test sinusoid."""
    super().setUp()
    self.amp = 0.75
    self.audio_len_sec = 4.0
    self.frequency = 440.0
    self.frame_rate = 250

  def validate_outputs(self, audio_features, expected_feature_lengths):
    for feat, expected_len in expected_feature_lengths.items():
      arr = audio_features[feat]
      try:
        self.assertLen(arr, expected_len)
      except AssertionError as e:
        raise AssertionError('%s feature: %s' % (e, feat))
      self.assertFalse(any(np.isinf(arr)))

  @parameterized.named_parameters(
      ('16k', 16000),
      ('24k', 24000),
      ('48k', 48000),
  )
  def test_compute_af_at_sample_rate(self, sample_rate):
    # Create test sinusoid
    x = np.linspace(0, self.audio_len_sec,
                    self.audio_len_sec * sample_rate)
    audio_sin = self.amp * (np.sin(2 * np.pi * self.frequency * x))
    # Compute audio features
    audio_features = compute_audio_features(audio_sin, sample_rate)

    expected_f0_and_loudness_length = self.audio_len_sec * self.frame_rate
    self.validate_outputs(
        audio_features,
        {
            'audio': self.audio_len_sec * sample_rate,
            'f0_hz': expected_f0_and_loudness_length,
            'f0_confidence': expected_f0_and_loudness_length,
            'loudness_db': expected_f0_and_loudness_length,
        })


if __name__ == '__main__':
  tf.test.main()
