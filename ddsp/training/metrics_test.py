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
"""Tests for ddsp.training.eval_util."""

from unittest import mock

from absl.testing import parameterized
from ddsp.spectral_ops_test import gen_np_batched_sinusoids
from ddsp.spectral_ops_test import gen_np_sinusoid
import ddsp.training.metrics as ddsp_metrics
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
    audio_features = ddsp_metrics.compute_audio_features(
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
      ddsp_metrics.compute_audio_features(
          audio_sin, sample_rate=sample_rate, frame_rate=self.frame_rate)


class MetricsObjectsTest(parameterized.TestCase, tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    """Create common default batch of noise and sinusoids."""
    super().setUpClass()
    cls.frequency = 440
    cls.sample_rate = 16000
    cls.frame_rate = 250
    cls.batch_size = 2
    cls.audio_len_sec = 0.25  # To make tests with f0 CREPE run in shorter time
    cls.amp = 0.8
    cls.batch_of_noise = cls.gen_batch_of_noise(cls.amp)
    cls.batch_of_noise_feats = cls.gen_batch_of_features(cls.batch_of_noise)
    cls.batch_of_sin = gen_np_batched_sinusoids(cls.frequency, cls.amp * 0.5,
                                                cls.sample_rate,
                                                cls.audio_len_sec,
                                                cls.batch_size)
    cls.batch_of_sin_feats = cls.gen_batch_of_features(cls.batch_of_sin)

  @classmethod
  def gen_batch_of_noise(cls, amp):
    noise_audio = np.random.uniform(
        low=-amp,
        high=amp,
        size=(1, int(cls.sample_rate * cls.audio_len_sec)))
    return np.tile(noise_audio, [cls.batch_size, 1])

  @classmethod
  def gen_batch_of_features(cls, batch_of_audio):
    batch_size = batch_of_audio.shape[0]
    audio = batch_of_audio[0]
    feats = ddsp_metrics.compute_audio_features(
        audio, sample_rate=cls.sample_rate, frame_rate=cls.frame_rate)
    for k, v in feats.items():
      feats[k] = np.tile(v[np.newaxis, :], [batch_size, 1])
    return feats

  def test_loudness_metrics_has_expected_values(self):
    loudness_metrics = ddsp_metrics.LoudnessMetrics(self.sample_rate,
                                                    self.frame_rate)
    # Dummy batch 1: known noise features vs. known noise audio
    loudness_metrics.update_state(self.batch_of_noise_feats,
                                  self.batch_of_noise)
    self.assertAllClose(loudness_metrics.metrics['loudness_db'].result(), 0)

    # Dummy batch 2: known noise features vs. quiet batch of sin audio
    loudness_metrics.update_state(self.batch_of_noise_feats, self.batch_of_sin)
    self.assertGreater(loudness_metrics.metrics['loudness_db'].result(), 0)

    loudness_metrics.flush(step=1)

  @mock.patch('ddsp.spectral_ops.compute_f0')
  def test_f0_crepe_metrics_has_expected_values(self, mock_compute_f0):
    """Test F0CrepeMetrics.

    F0CrepeMetrics makes an expensive call to compute_f0 (which in turn calls
    CREPE) for every generated example during update_state. To avoid this, we
    mock out compute_f0 and replace the return values (via side_effect) with
    precomputed f0_hz and confidence values.

    Args:
      mock_compute_f0: The mock object for compute_f0, automatically injected
        by mock.patch.
    """
    f0_crepe_metrics = ddsp_metrics.F0CrepeMetrics(self.sample_rate,
                                                   self.frame_rate)
    # Batch 1: correct f0
    crepe_f0 = self.batch_of_sin_feats['f0_hz']
    crepe_conf = np.ones_like(crepe_f0)
    mock_compute_f0.side_effect = zip(crepe_f0, crepe_conf)
    f0_crepe_metrics.update_state(self.batch_of_sin_feats, self.batch_of_sin)
    self.assertAllClose(f0_crepe_metrics.metrics['f0_dist'].result(), 0)
    self.assertAllClose(
        f0_crepe_metrics.metrics['outlier_ratio'].result(), 0)

    # Batch 2: incorrect f0
    crepe_f0 = self.batch_of_sin_feats['f0_hz'] * 2
    crepe_conf = np.ones_like(crepe_f0)
    mock_compute_f0.side_effect = zip(crepe_f0, crepe_conf)
    f0_crepe_metrics.update_state(self.batch_of_sin_feats, self.batch_of_sin)

    self.assertGreater(f0_crepe_metrics.metrics['f0_dist'].result(), 0)
    self.assertAllClose(
        f0_crepe_metrics.metrics['outlier_ratio'].result(), 0)

    # Batch 3: low crepe confidence
    crepe_f0 = np.zeros_like(self.batch_of_sin_feats['f0_hz'])
    crepe_conf = np.ones_like(crepe_f0)
    mock_compute_f0.side_effect = zip(crepe_f0, crepe_conf)
    f0_crepe_metrics.update_state(self.batch_of_sin_feats, self.batch_of_noise)
    self.assertGreater(f0_crepe_metrics.metrics['f0_dist'].result(), 0)
    self.assertGreater(
        f0_crepe_metrics.metrics['outlier_ratio'].result(), 0)

    f0_crepe_metrics.flush(step=1)

  def test_f0_metrics_has_expected_values(self):
    f0_metrics = ddsp_metrics.F0Metrics(self.sample_rate, self.frame_rate)
    # Batch 1: known sin features vs. batch of known sin f0_hz
    f0_metrics.update_state(self.batch_of_sin_feats,
                            self.batch_of_sin_feats['f0_hz'])
    self.assertAllClose(f0_metrics.metrics['f0_dist'].result(), 0)

    # Batch 2: known sin features vs. batch of f0_hz at different f0
    f0_metrics.update_state(self.batch_of_sin_feats,
                            self.batch_of_sin_feats['f0_hz'] * 3)
    self.assertGreater(f0_metrics.metrics['f0_dist'].result(), 0)

    f0_metrics.flush(step=1)

  def test_f0_metrics_resamples_f0_hz_predictions_to_expected_length(self):
    f0_metrics = ddsp_metrics.F0Metrics(self.sample_rate, self.frame_rate)
    expected_len = self.frame_rate * self.audio_len_sec
    shorter_len = int(expected_len * 0.8)

    # Batch 1: known sin features vs. batch of shorter f0_hz at different f0
    f0_metrics.update_state(
        self.batch_of_sin_feats,
        3 * self.batch_of_sin_feats['f0_hz'][:, :shorter_len])
    self.assertGreater(f0_metrics.metrics['f0_dist'].result(), 0)

    f0_metrics.flush(step=1)

  def test_rpa_has_expected_values_exact_match(self):
    rpa = ddsp_metrics.F0Metrics(self.sample_rate, self.frame_rate)
    f0 = self.batch_of_sin_feats['f0_hz']
    rpa.update_state(self.batch_of_sin_feats, f0)
    self.assertEqual(rpa.metrics['raw_pitch_accuracy'].result(), 1.0)
    self.assertEqual(rpa.metrics['raw_chroma_accuracy'].result(), 1.0)

  def test_rpa_has_expected_values_octave_error(self):
    rpa = ddsp_metrics.F0Metrics(self.sample_rate, self.frame_rate)
    f0 = self.batch_of_sin_feats['f0_hz']
    rpa.update_state(self.batch_of_sin_feats, f0 * 2)
    self.assertEqual(rpa.metrics['raw_pitch_accuracy'].result(), 0.0)
    self.assertEqual(rpa.metrics['raw_chroma_accuracy'].result(), 1.0)

  def test_rpa_has_expected_values_error_within_threshold(self):
    rpa = ddsp_metrics.F0Metrics(self.sample_rate, self.frame_rate)
    f0 = self.batch_of_sin_feats['f0_hz']
    rpa.update_state(self.batch_of_sin_feats, f0 + 10)
    self.assertEqual(rpa.metrics['raw_pitch_accuracy'].result(), 1.0)
    self.assertEqual(rpa.metrics['raw_chroma_accuracy'].result(), 1.0)

  def test_rpa_has_expected_values_error_outside_threshold(self):
    rpa = ddsp_metrics.F0Metrics(self.sample_rate, self.frame_rate)
    f0 = self.batch_of_sin_feats['f0_hz']
    rpa.update_state(self.batch_of_sin_feats, f0 + 220)
    self.assertEqual(rpa.metrics['raw_pitch_accuracy'].result(), 0.0)
    self.assertEqual(rpa.metrics['raw_chroma_accuracy'].result(), 0.0)


if __name__ == '__main__':
  tf.test.main()
