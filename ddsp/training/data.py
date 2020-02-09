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
"""Library of functions to help loading data."""

from absl import logging
import gin
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

_AUTOTUNE = tf.data.experimental.AUTOTUNE


# ---------- Base Class --------------------------------------------------------
class DataProvider(object):
  """Base class for returning a dataset."""

  def get_dataset(self, shuffle):
    """A method that returns a tf.data.Dataset."""
    raise NotImplementedError

  def get_batch(self, batch_size, shuffle=True, repeats=-1):
    """Read dataset.

    Args:
      batch_size: Size of batch.
      shuffle: Whether to shuffle the examples.
      repeats: Number of times to repeat dataset. -1 for endless repeats.

    Returns:
      A batched tf.data.Dataset.
    """
    dataset = self.get_dataset(shuffle)
    dataset = dataset.repeat(repeats)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=_AUTOTUNE)
    return dataset


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
    self._name = name
    self._split = split
    self._data_dir = data_dir

  def get_dataset(self, shuffle=True):
    """Read dataset.

    Args:
      shuffle: Whether to shuffle the input files.

    Returns:
      dataset: A tf.data.Dataset that reads from TFDS.
    """
    return tfds.load(
        self._name,
        data_dir=self._data_dir,
        split=self._split,
        shuffle_files=shuffle,
        download=False)


@gin.register
class NSynthTfds(TfdsProvider):
  """Parses features in the TFDS NSynth dataset.

  If running on Cloud, it is recommended you set `data_dir` to
  'gs://tfds-data/datasets' to avoid unnecessary downloads.
  """

  def __init__(self,
               name='nsynth/gansynth_subset.f0_and_loudness:2.3.0',
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
      logging.warning(
          'Using public TFDS GCS bucket to load NSynth. If not running on '
          'GCP, this will be very slow, and it is recommended you prepare '
          'the dataset locally with TFDS and set the data_dir appropriately.')
    super().__init__(name, split, data_dir)

  def get_dataset(self, shuffle=True):
    """Returns dataset with slight restructuring of feature dictionary."""
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
          'f0_hz':
              ex['f0']['hz'],
          'f0_confidence':
              ex['f0']['confidence'],
          'loudness_db':
              ex['loudness']['db'],
      }
    dataset = super().get_dataset(shuffle)
    dataset = dataset.map(preprocess_ex, num_parallel_calls=_AUTOTUNE)
    return dataset


@gin.register
class TFRecordProvider(DataProvider):
  """Class for reading TFRecord and returning a dataset."""

  def __init__(self,
               file_pattern=None,
               example_secs=4,
               sample_rate=16000,
               frame_rate=250):
    """TFRecordProvider constructor."""
    self._file_pattern = file_pattern or self.default_file_pattern
    self._audio_length = example_secs * sample_rate
    self._feature_length = example_secs * frame_rate

  @property
  def default_file_pattern(self):
    """Used if file_pattern is not provided to constructor."""
    raise NotImplementedError(
        'You must pass a "file_pattern" argument to the constructor or '
        'choose a FileDataProvider with a default_file_pattern.')

  def get_dataset(self, shuffle=True):
    """Read dataset.

    Args:
      shuffle: Whether to shuffle the files.

    Returns:
      dataset: A tf.dataset that reads from the TFRecord.
    """
    def parse_tfexample(record):
      return tf.io.parse_single_example(record, self.features_dict)

    filenames = tf.data.Dataset.list_files(self._file_pattern, shuffle=shuffle)
    dataset = filenames.interleave(
        map_func=tf.data.TFRecordDataset,
        cycle_length=40,
        num_parallel_calls=_AUTOTUNE)
    dataset = dataset.map(parse_tfexample, num_parallel_calls=_AUTOTUNE)
    return dataset

  @property
  def features_dict(self):
    """Dictionary of features to read from dataset."""
    return {
        'audio':
            tf.io.FixedLenFeature([self._audio_length], dtype=tf.float32),
        'f0_hz':
            tf.io.FixedLenFeature([self._feature_length], dtype=tf.float32),
        'f0_confidence':
            tf.io.FixedLenFeature([self._feature_length], dtype=tf.float32),
        'loudness_db':
            tf.io.FixedLenFeature([self._feature_length], dtype=tf.float32),
    }


