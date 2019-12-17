# Copyright 2019 The DDSP Authors.
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

"""Tests for ddsp.training.data_preparation.prepare_tfrecord_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from absl import flags
from absl.testing import absltest
from ddsp.training.data_preparation import prepare_tfrecord_lib
import librosa
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.enable_eager_execution()


class ProcessTaskBeamTest(absltest.TestCase):

  def get_tempdir(self):
    try:
      flags.FLAGS.test_tmpdir
    except flags.UnparsedFlagAccessError:
      # Need to initialize flags when running `pytest`.
      flags.FLAGS(sys.argv)
    return self.create_tempdir().full_path

  def setUp(self):
    super(ProcessTaskBeamTest, self).setUp()
    self.test_dir = self.get_tempdir()

    # Write test wav file.
    self.wav_sr = 22050
    self.wav_secs = 2.1
    self.wav_path = os.path.join(self.test_dir, 'test.wav')
    librosa.output.write_wav(
        self.wav_path,
        np.random.rand(int(self.wav_sr * self.wav_secs)),
        sr=self.wav_sr)

  def parse_tfrecord(self, path):
    return [tf.train.Example.FromString(record.numpy()) for record in
            tf.data.TFRecordDataset(os.path.join(self.test_dir, path))]

  def validate_outputs(self, expected_num_examples, expected_feature_lengths):
    all_examples = (
        self.parse_tfrecord('output.tfrecord-00000-of-00002') +
        self.parse_tfrecord('output.tfrecord-00001-of-00002'))

    self.assertLen(all_examples, expected_num_examples)
    for ex in all_examples:
      self.assertCountEqual(expected_feature_lengths, ex.features.feature)
      for feat, expected_len in expected_feature_lengths.items():
        arr = ex.features.feature[feat].float_list.value
        try:
          self.assertLen(arr, expected_len)
        except AssertionError as e:
          raise AssertionError('%s feature: %s' % (e, feat))
        self.assertFalse(any(np.isinf(arr)))

  def test_prepare_tfrecord(self):
    audio_rate = 16000
    f0_and_loudness_rate = 250
    window_secs = 1
    hop_secs = 0.5
    prepare_tfrecord_lib.prepare_tfrecord(
        [self.wav_path],
        os.path.join(self.test_dir, 'output.tfrecord'),
        num_shards=2,
        audio_rate=audio_rate,
        f0_and_loudness_rate=f0_and_loudness_rate,
        window_size=window_secs * audio_rate,
        hop_size=int(hop_secs * audio_rate))

    expected_f0_and_loudness_length = window_secs * f0_and_loudness_rate
    self.validate_outputs(
        4,
        {
            'audio': window_secs * audio_rate,
            'f0': expected_f0_and_loudness_length,
            'f0_confidence': expected_f0_and_loudness_length,
            'loudness': expected_f0_and_loudness_length,
            'loudness_uncentered': expected_f0_and_loudness_length,
        })

  def test_prepare_tfrecord_no_split(self):
    audio_rate = 16000
    f0_and_loudness_rate = 250
    prepare_tfrecord_lib.prepare_tfrecord(
        [self.wav_path],
        os.path.join(self.test_dir, 'output.tfrecord'),
        num_shards=2,
        audio_rate=audio_rate,
        f0_and_loudness_rate=f0_and_loudness_rate,
        window_size=None)

    expected_f0_and_loudness_length = self.wav_secs * f0_and_loudness_rate
    self.validate_outputs(
        1,
        {
            'audio': self.wav_secs * audio_rate,
            'f0': expected_f0_and_loudness_length,
            'f0_confidence': expected_f0_and_loudness_length,
            'loudness': expected_f0_and_loudness_length,
            'loudness_uncentered': expected_f0_and_loudness_length,
        })

  def test_prepare_tfrecord_no_f0_and_loudness(self):
    audio_rate = 16000
    prepare_tfrecord_lib.prepare_tfrecord(
        [self.wav_path],
        os.path.join(self.test_dir, 'output.tfrecord'),
        num_shards=2,
        audio_rate=audio_rate,
        f0_and_loudness_rate=None,
        window_size=None)

    self.validate_outputs(1, {'audio': self.wav_secs * audio_rate})

if __name__ == '__main__':
  absltest.main()
