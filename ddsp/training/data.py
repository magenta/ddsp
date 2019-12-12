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

from absl import logging
import gin
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds


# ---------- Base Class --------------------------------------------------------
class DataProvider(object):
  """Base class for returning a dataset."""

  def get_dataset(self, shuffle):
    """A method that returns a tf.data.Dataset."""
    raise NotImplementedError

  def get_preprocess_fn(self):
    """A method that returns a per-record preprocess function.

    Defaults to a no-op if not overriden.

    Returns:
      A callable mapping a record to a preprocessed record.
    """
    return lambda x: x

  def get_batch(self, batch_size, shuffle=True, repeats=-1):
    """Read dataset.

    Args:
      batch_size: Size of batch.
      shuffle: Whether to shuffle the examples.
      repeats: Number of times to repeat dataset. -1 for endless repeats.

    Returns:
      A batched tf.data.Dataset.
    """
    autotune = tf.data.experimental.AUTOTUNE

    dataset = self.get_dataset(shuffle)
    dataset = dataset.map(self.get_preprocess_fn(), num_parallel_calls=autotune)
    if shuffle:
      dataset = dataset.shuffle(buffer_size=10 * 1000)
    dataset = dataset.repeat(repeats)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=autotune)
    return dataset

  def get_input_fn(self, shuffle=True, repeats=-1):
    """Wrapper to make get_batch() compatible with tf.Estimator.

    Args:
      shuffle: Whether to shuffle the examples.
      repeats: Number of times to repeat dataset. -1 for endless repeats.

    Returns:
      An input_fn() for a tf.Estimator. Function takes in a dictionary 'params'
        as its first argument, with key 'batch_size', and returns a dataset
        that has a tuple of examples for each entry.
    """

    def input_fn(params):
      batch_size = params['batch_size']
      dataset = self.get_batch(
          batch_size=batch_size, shuffle=shuffle, repeats=repeats)
      dataset = dataset.map(
          lambda ex: (ex, ex), num_parallel_calls=tf.data.experimental.AUTOTUNE)
      return dataset

    return input_fn


class TfdsProvider(DataProvider):
  """Base class for reading datasets from TensorFlow Datasets (TFDS)."""

  def __init__(self, name, split, data_dir):
    """TfdsProvider constructor.

    Args:
      name: TFDS dataset name (with optional config and version).
      split: Dataset split to use of the TFDS dataset.
      data_dir: The directory to read TFDS datasets from. Defaults to
        "~/tensorflow_datasets".
    """
    self.name = name
    self.split = split
    self.data_dir = data_dir

  def get_dataset(self, shuffle=True):
    """Read dataset.

    Args:
      shuffle: Whether to shuffle the input files.

    Returns:
      dataset: A tf.data.Dataset that reads from TFDS.
    """
    return tfds.load(
        self.name,
        data_dir=self.data_dir,
        split=self.split,
        shuffle_files=shuffle,
        download=False)


class TFRecordProvider(DataProvider):
  """Base class for reading files and returning a dataset."""

  def __init__(self, file_pattern=None):
    """Specifyies the regular expression of the TFRecord files."""
    self.file_pattern = file_pattern or self.default_file_pattern

  @property
  def default_file_pattern(self):
    """Used if file_pattern is not provided to constructor."""
    raise NotImplementedError(
        'You must pass a "file_pattern" argument to the constructor or '
        'choose a FileDataProvider with a default_file_pattern.')

  @property
  def features_dict(self):
    """Dictionary of features to read from dataset."""
    raise NotImplementedError

  @property
  def file_reader(self):
    """Which type of dataset from which to read."""
    return tf.data.TFRecordDataset

  def get_preprocess_fn(self):

    def parse_tfexample(*args):
      """Only parse the tf.Example string."""
      if len(args) == 2:
        record = args[1]
      else:
        record = args
      return tf.parse_single_example(record, self.features_dict)

    return parse_tfexample

  def get_dataset(self, shuffle=True):
    """Read dataset.

    Args:
      shuffle: Whether to shuffle the files.

    Returns:
      dataset: A tf.dataset that reads from the TFRecord.
    """
    filenames = tf.data.Dataset.list_files(self.file_pattern, shuffle=shuffle)
    dataset = filenames.interleave(
        map_func=self.file_reader,
        cycle_length=40,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


# ---------- Different Dataset Types -------------------------------------------
@gin.configurable()
class NSynthTfds(TfdsProvider):
  """Parses features in the TFDS NSynth dataset.

  If running on Cloud, it is recommended you set `data_dir` to
  'gs://tfds-data/datasets' to avoid unnecessary downloads.
  """

  def __init__(self,
               name='nsynth/gansynth_subset.f0_and_loudness',
               split='train',
               data_dir='gs://tfds-data/datasets'):
    """TfdsProvider constructor.

    Args:
      name: TFDS dataset name (with optional config and version).
      split: Dataset split to use of the TFDS dataset.
      data_dir: The directory to read the prepared NSynth dataset from. Defaults
        to the public TFDS GCS bucket.
    """
    if data_dir == 'gs://tfds-data/datasets':
      logging.warn(
          'Using public TFDS GCS bucket to load NSynth. If not running on '
          'cloud, this will be very slow, and it is recommended you prepare '
          'the dataset locally with TFDS and set the data_dir appropriately.')
    super(NSynthTfds, self).__init__(name, split, data_dir)

  def get_preprocess_fn(self):
    loudness_stats = tfds.builder(self.name).info.metadata

    def preprocess_ex(ex):
      return {
          'pitch':
              ex['pitch'],
          'audio':
              ex['audio'],
          'instrument_source':
              ex['instrument']['source'],
          'instrument_family':
              ex['instrument']['family'],
          'instrument':
              ex['instrument']['label'],
          'f0':
              ex['f0']['hz'],
          'f0_confidence':
              ex['f0']['confidence'],
          'loudness': (
              (ex['loudness']['db'] - loudness_stats['loudness_db_mean']) /
              tf.math.sqrt(loudness_stats['loudness_db_variance'])),
      }

    return preprocess_ex


@gin.configurable
class NSynthTFRecord(TFRecordProvider):
  """Parses features in the TFRecord NSynth dataset."""

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
class SoloInstrument(TFRecordProvider):
  """Parses features in a solo instrument dataset."""

  @property
  def features_dict(self):
    return {
        'audio': tf.FixedLenFeature([64000], dtype=tf.float32),
        'f0': tf.FixedLenFeature([1001], dtype=tf.float32),
        'f0_confidence': tf.FixedLenFeature([1001], dtype=tf.float32),
        'loudness': tf.FixedLenFeature([1001], dtype=tf.float32),
    }


