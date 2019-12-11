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

  def get_dataset(self, split, shuffle, **kwargs):
    """A method that return a dataset."""
    raise NotImplementedError

  def get_preprocess_fn(self):
    """A method that returns a per-record preprocess function.

    Defaults to a no-op if not overriden.

    Returns:
      A callable mapping a record to a preprocessed record.
    """
    return lambda x: x

  def get_batch(self, batch_size, shuffle=True, repeats=-1, **dataset_kwargs):
    """Read dataset.

    Args:
      batch_size: Size of batch.
      shuffle: Whether to shuffle the examples.
      repeats: Number of times to repeat dataset. -1 for endless repeats.
      **dataset_kwargs: Additonal keyword arguments for the DataProvider.

    Returns:
      A batched tf.dataset.
    """
    autotune = tf.data.experimental.AUTOTUNE

    dataset = self.get_dataset(shuffle, **dataset_kwargs)
    dataset = dataset.map(self.get_preprocess_fn(), num_parallel_calls=autotune)
    if shuffle:
      dataset = dataset.shuffle(buffer_size=10 * 1000)
    dataset = dataset.repeat(repeats)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=autotune)
    return dataset

  def get_input_fn(self, shuffle=True, repeats=-1, **dataset_kwargs):
    """Wrapper to make compatible with tf.Estimator.

    It seems that @gin.configurable overwites the function names, which violates
    tf.TPUEstimator's interface that requires an arg to be named 'params'. This
    wrapper function gets around that collision.

    Args:
      shuffle: Whether to shuffle the examples.
      repeats: Number of times to repeat dataset. -1 for endless repeats.
      **dataset_kwargs: Additonal keyword arguments for the dataset.

    Returns:
      A tf.dataset.
    """
    def input_fn(params):
      batch_size = params['batch_size']
      dataset = self.get_batch(
          batch_size=batch_size, shuffle=shuffle, repeats=repeats,
          **dataset_kwargs)
      dataset = dataset.map(
          lambda ex: (ex, ex),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
      return dataset

    return input_fn


class TFRecordProvider(DataProvider):
  """Base class for reading files and returning a dataset."""

  @property
  def default_file_pattern(self):
    """File pattern dict bound to class, used if no file_pattern provided."""
    raise NotImplementedError(
        'You must either specify a --file_pattern flag with a regular '
        'expression for the dataset file paths or choose a FileDataProvider '
        'with default file patterns.')

  @property
  def features_dict(self):
    """Dictionary of features to read from dataset."""
    raise NotImplementedError

  @property
  def file_reader(self):
    """Dataset file reader."""
    return tf.data.TFRecordDataset

  def get_preprocess_fn(self):
    def parse_tfexample(*args):
      if len(args) == 2:
        record = args[1]
      else:
        record = args
      return tf.parse_single_example(record, self.features_dict)

    return parse_tfexample

  @gin.configurable(module='data.TFRecordProvider')
  def get_dataset(self, shuffle=True, file_pattern=None):
    """Read dataset.

    Args:
      shuffle: Whether to shuffle the files.
      file_pattern: Regex pattern to all the tfrecord files. Defaults to
          `self.default_file_pattern`.

    Returns:
      dataset: A tf.dataset that reads from the TFRecord.
    """

    file_pattern = file_pattern or self.default_file_pattern
    filenames = tf.data.Dataset.list_files(file_pattern,
                                           shuffle=shuffle)

    dataset = filenames.interleave(
        map_func=self.file_reader,
        cycle_length=40,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


class TfdsProvider(DataProvider):
  """Base class for reading datasets from TensorFlow Datasets (TFDS)."""

  def __init__(self, data_dir=None):
    """TfdsProvider constructor.

    Args:
      data_dir: The directory to read/write data when downloading and preparing
        new TFDS datasets. Defaults to "~/tensorflow_datasets".
    """
    self._tfds_dir = data_dir

  @property
  def default_tfds_name(self):
    """TFDS name bound to class, used if no `tfds_name` provided.

    May optionally include the config name and version number.
    """
    raise NotImplementedError(
        'You must either specify a --tfds_name flag or choose a '
        'TfdsDataProvider with a default dataset name.')

  @property
  def default_tfds_split(self):
    """TFDS split bound to class, used if no `tfds_split` provided."""
    raise NotImplementedError(
        'You must either specify a --tfds_split flag or choose a '
        'TfdsDataProvider with a default split.')

  @property
  def default_tfds_data_dir(self):
    """TFDS data dir bound to class, used if no `tfds_data_dir` provided."""
    raise NotImplementedError(
        'You must either specify a --tfds_data_dir flag or choose a '
        'TfdsDataProvider with a default data directory.')

  @gin.configurable(module='data.TfdsProvider')
  def get_dataset(
      self, shuffle=True, tfds_name=None, tfds_split=gin.REQUIRED,
      tfds_data_dir=None):
    """Read dataset.

    Args:
      shuffle: Whether to shuffle the input files.
      tfds_name: TFDS dataset name (with optional config and version). Defaults
        to `self.default_tfds_name`.
      tfds_split: The name of the split to use. Defaults to
        `self.default_tfds_split`.
      tfds_data_dir: The directory to read TFDS data from. Defaults to
        `self.default_tfds_data_dir`.

    Returns:
      dataset: A tf.dataset that reads from TFDS.
    """
    logging.info(tfds_split)
    tfds_name = tfds_name or self.default_tfds_name
    tfds_split = tfds_split or self.default_tfds_split
    tfds_data_dir = tfds_data_dir or self.default_tfds_data_dir
    return tfds.load(
        tfds_name,
        data_dir=tfds_data_dir,
        split=tfds_split,
        shuffle_files=shuffle,
        download=False)


# ---------- Different Dataset Types -------------------------------------------
@gin.configurable()
class NSynthTfds(TfdsProvider):
  """Parses features in the TFDS NSynth dataset.

  If running on Cloud, it is recommended you set `data_dir` to
  'gs://tfds-data/datasets' to avoid unnecessary downloads.
  """

  @property
  def default_tfds_name(self):
    return 'nsynth/gansynth_subset.f0_and_loudness'

  @property
  def default_tfds_data_dir(self):
    logging.warn(
        'Using public TFDS GCS bucket load NSynth. If not running on cloud, it '
        'is recommended you prepare the dataset locally with TFDS and set '
        '`--tfds_data_dir` appropriately.')
    default_dir = 'gs://tfds-data/datasets'
    return default_dir

  def get_preprocess_fn(self):
    loudness_stats = tfds.builder(
        'nsynth/gansynth_subset.f0_and_loudness').info.metadata
    def preprocess_ex(ex):
      return {
          'pitch': ex['pitch'],
          'audio': ex['audio'],
          'instrument_source': ex['instrument']['source'],
          'instrument_family': ex['instrument']['family'],
          'instrument': ex['instrument']['label'],
          'f0': ex['f0']['hz'],
          'f0_confidence': ex['f0']['confidence'],
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
