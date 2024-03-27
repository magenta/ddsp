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

"""Tests for ddsp.training.data_preparation.prepare_tfrecord_lib."""

import os
import sys

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from ddsp import spectral_ops
from ddsp.training.data_preparation import prepare_tfrecord_lib
import numpy as np
import scipy.io.wavfile
import tensorflow.compat.v2 as tf

CREPE_SAMPLE_RATE = spectral_ops.CREPE_SAMPLE_RATE


class PrepareTFRecordBeamTest(parameterized.TestCase):

  def get_tempdir(self):
    try:
      flags.FLAGS.test_tmpdir
    except flags.UnparsedFlagAccessError:
      # Need to initialize flags when running `pytest`.
      flags.FLAGS(sys.argv)
    return self.create_tempdir().full_path

  def setUp(self):
    super().setUp()
    self.test_dir = self.get_tempdir()

    # Write test wav file.
    self.wav_sr = 22050
    self.wav_secs = 0.5
    self.wav_path = os.path.join(self.test_dir, 'test.wav')
    scipy.io.wavfile.write(
        self.wav_path,
        self.wav_sr,
        np.random.randint(
            np.iinfo(np.int16).min, np.iinfo(np.int16).max,
            size=int(self.wav_sr * self.wav_secs), dtype=np.int16))

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
          raise AssertionError('feature: %s' % feat) from e
        self.assertFalse(any(np.isinf(arr)))

  def get_expected_length(self, input_length, frame_rate, center=False):
    sample_rate = 16000  # Features at CREPE_SAMPLE_RATE.
    frame_size = 1024  # Unused for this calculation.
    hop_size = sample_rate // frame_rate
    padding = 'center' if center else 'same'
    n_frames, _ = spectral_ops.get_framed_lengths(
        input_length, frame_size, hop_size, padding)
    return n_frames

  @staticmethod
  def get_n_per_chunk(chunk_length, example_secs, hop_secs):
    """Convenience function to calculate number examples from striding."""
    n = (chunk_length - example_secs) / hop_secs
    # Deal with limited float precision that causes (.3 / .1) = 2.9999....
    return int(np.floor(np.round(n, decimals=3))) + 1

  @parameterized.named_parameters(
      ('chunk_and_split', 0.3, 0.2),
      ('no_chunk', None, 0.2),
      ('no_split', 0.3, None),
      ('no_chunk_no_split', None, None),
  )
  def test_prepare_tfrecord(self, chunk_secs, example_secs):
    sample_rate = 16000
    frame_rate = 250
    hop_secs = 0.1

    # Calculate expected batch size.
    if example_secs:
      length = chunk_secs if chunk_secs else self.wav_secs
      n_per_chunk = self.get_n_per_chunk(length, example_secs, hop_secs)
    else:
      n_per_chunk = 1

    n_chunks = int(np.ceil(self.wav_secs / chunk_secs)) if chunk_secs else 1
    expected_n_batch = n_per_chunk * n_chunks
    print('n_chunks, n_per_chunk, chunk_secs, example_secs',
          n_chunks, n_per_chunk, chunk_secs, example_secs)

    # Calculate expected lengths.
    if example_secs:
      length = example_secs
    elif chunk_secs:
      length = chunk_secs
    else:
      length = self.wav_secs

    expected_n_t = int(length * sample_rate)
    expected_n_frames = self.get_expected_length(expected_n_t, frame_rate)

    # Make the actual records.
    prepare_tfrecord_lib.prepare_tfrecord(
        [self.wav_path],
        os.path.join(self.test_dir, 'output.tfrecord'),
        num_shards=2,
        sample_rate=sample_rate,
        frame_rate=frame_rate,
        example_secs=example_secs,
        hop_secs=hop_secs,
        chunk_secs=chunk_secs,
        center=False)

    self.validate_outputs(
        expected_n_batch,
        {
            'audio': expected_n_t,
            'audio_16k': expected_n_t,
            'f0_hz': expected_n_frames,
            'f0_confidence': expected_n_frames,
            'loudness_db': expected_n_frames,
        })

  @parameterized.named_parameters(('no_center', False), ('center', True))
  def test_centering(self, center):
    frame_rate = 250
    sample_rate = 16000
    example_secs = 0.3
    hop_secs = 0.1
    n_batch = self.get_n_per_chunk(self.wav_secs, example_secs, hop_secs)
    prepare_tfrecord_lib.prepare_tfrecord(
        [self.wav_path],
        os.path.join(self.test_dir, 'output.tfrecord'),
        num_shards=2,
        sample_rate=sample_rate,
        frame_rate=frame_rate,
        example_secs=example_secs,
        hop_secs=hop_secs,
        center=center,
        chunk_secs=None)

    n_t = int(example_secs * sample_rate)
    n_frames = self.get_expected_length(n_t, frame_rate, center)
    n_expected_frames = 76 if center else 75  # (250 * 0.3) [+1].
    self.assertEqual(n_frames, n_expected_frames)
    self.validate_outputs(
        n_batch, {
            'audio': n_t,
            'audio_16k': n_t,
            'f0_hz': n_frames,
            'f0_confidence': n_frames,
            'loudness_db': n_frames,
        })

  @parameterized.named_parameters(
      ('16kHz', 16000),
      ('32kHz', 32000),
      ('48kHz', 48000))
  def test_sample_rate(self, sample_rate):
    frame_rate = 250
    example_secs = 0.3
    hop_secs = 0.1
    center = True
    n_batch = self.get_n_per_chunk(self.wav_secs, example_secs, hop_secs)
    prepare_tfrecord_lib.prepare_tfrecord(
        [self.wav_path],
        os.path.join(self.test_dir, 'output.tfrecord'),
        num_shards=2,
        sample_rate=sample_rate,
        frame_rate=frame_rate,
        example_secs=example_secs,
        hop_secs=hop_secs,
        center=center,
        chunk_secs=None)

    n_t = int(example_secs * sample_rate)
    n_t_16k = int(example_secs * CREPE_SAMPLE_RATE)
    n_frames = self.get_expected_length(n_t_16k, frame_rate, center)
    n_expected_frames = 76  # (250 * 0.3) + 1.
    self.assertEqual(n_frames, n_expected_frames)
    self.validate_outputs(
        n_batch, {
            'audio': n_t,
            'audio_16k': n_t_16k,
            'f0_hz': n_frames,
            'f0_confidence': n_frames,
            'loudness_db': n_frames,
        })

  @parameterized.named_parameters(('16kHz', 16000), ('44.1kHz', 44100),
                                  ('48kHz', 48000))
  def test_audio_only(self, sample_rate):
    prepare_tfrecord_lib.prepare_tfrecord(
        [self.wav_path],
        os.path.join(self.test_dir, 'output.tfrecord'),
        num_shards=2,
        sample_rate=sample_rate,
        frame_rate=None,
        example_secs=None,
        chunk_secs=None)

    self.validate_outputs(
        1, {
            'audio': int(self.wav_secs * sample_rate),
            'audio_16k': int(self.wav_secs * CREPE_SAMPLE_RATE),
        })


if __name__ == '__main__':
  absltest.main()
