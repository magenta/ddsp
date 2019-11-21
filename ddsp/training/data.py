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

# Lint as: python3
"""Library of functions to help loading data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import gin
import tensorflow as tf  # tf


# ---------- Base Class --------------------------------------------------------
class FileReader(object):
  """Base class for reading files and returning a dataset."""

  @property
  def default_file_pattern(self):
    """File pattern bound to class, used if no file_pattern provided."""
    raise NotImplementedError(
        'You must either specify a --file_pattern flag with a regular '
        'expression for the dataset file paths or choose a FileReader with a '
        'default file pattern.')

  @property
  def features_dict(self):
    """Dictionary of features to read from dataset."""
    raise NotImplementedError

  @property
  def map_function(self):
    """Mapping function based on dataset type."""
    return tf.data.TFRecordDataset

  def get_batch(self, params, shuffle=True, repeats=-1, file_pattern=None):
    """Read dataset.

    Args:
      params: Dictionary specifying hyperparameters. Called 'params' here
        because that is the interface that TPUEstimator expects.
      shuffle: Shuffle the examples.
      repeats: Number of times to repeat dataset. -1 for endless repeats.
      file_pattern: Regex pattern to all the tfrecord files. Defaults to
          self.default_file_pattern.

    Returns:
      dataset: A tf.dataset that reads from the sstable.
    """
    file_pattern = file_pattern or self.default_file_pattern
    batch_size = params['batch_size']
    autotune = tf.data.experimental.AUTOTUNE

    def parse_tfrecord(unused_key, record):
      example = tf.parse_single_example(record, self.features_dict)
      return (example, example)  # Return tuple for estimator api (data, label).

    filenames = tf.data.Dataset.list_files(file_pattern,
                                           shuffle=shuffle)

    dataset = filenames.interleave(map_func=self.map_function,
                                   cycle_length=40,
                                   num_parallel_calls=autotune)
    if shuffle:
      dataset = dataset.shuffle(buffer_size=10 * 1000)
    dataset = dataset.map(parse_tfrecord)
    dataset = dataset.repeat(repeats)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=autotune)
    return dataset

  def get_input_fn(self,
                   shuffle=True,
                   repeats=-1,
                   file_pattern=None):
    """Wrapper to make compatible with tf.Estimator.

    It seems that @gin.configurable overwites the function names, which violates
    tf.TPUEstimator's interface that requires an arg to be named 'params'. This
    wrapper function gets around that collision.

    Args:
      shuffle: Shuffle the examples.
      repeats: Number of times to repeat dataset. -1 for endless repeats.
      file_pattern: Regex pattern to all the tfrecord files. Overides
          FileReader.default_file_pattern.

    Returns:
      A tf.dataset that reads parsed files.
    """
    return functools.partial(self.get_batch,
                             shuffle=shuffle,
                             repeats=repeats,
                             file_pattern=file_pattern)


# ---------- Different Dataset Types -------------------------------------------
@gin.configurable
class NSynth(FileReader):
  """Parses features in the NSynth dataset."""

  @property
  def features_dict(self):
    return {
        'pitch': tf.FixedLenFeature([1], dtype=tf.int64),
        'audio': tf.FixedLenFeature([64000], dtype=tf.float32),
        'qualities': tf.FixedLenFeature([10], dtype=tf.int64),
        'instrument_source': tf.FixedLenFeature([1], dtype=tf.int64),
        'instrument_family': tf.FixedLenFeature([1], dtype=tf.int64),
        'instrument': tf.FixedLenFeature([1], dtype=tf.int64),
        'f0': tf.FixedLenFeature([1001], dtype=tf.float32),
        'f0_confidence': tf.FixedLenFeature([1001], dtype=tf.float32),
        'loudness': tf.FixedLenFeature([1001], dtype=tf.float32),
        'harmonicity': tf.FixedLenFeature([1001], dtype=tf.float32),
    }


@gin.configurable
class SoloInstrument(FileReader):
  """Parses features in a solo instrument dataset."""

  @property
  def features_dict(self):
    return {
        'audio': tf.FixedLenFeature([64000], dtype=tf.float32),
        'f0': tf.FixedLenFeature([1001], dtype=tf.float32),
        'f0_confidence': tf.FixedLenFeature([1001], dtype=tf.float32),
        'loudness': tf.FixedLenFeature([1001], dtype=tf.float32),
    }
