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

"""Library of functions to help loading data."""
import os

from absl import logging
from ddsp.spectral_ops import CREPE_FRAME_SIZE
from ddsp.spectral_ops import CREPE_SAMPLE_RATE
from ddsp.spectral_ops import get_framed_lengths
import gin
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


_AUTOTUNE = tf.data.experimental.AUTOTUNE


# ---------- Base Class --------------------------------------------------------
class DataProvider(object):
  """Base class for returning a dataset."""

  def __init__(self, sample_rate, frame_rate):
    """DataProvider constructor.

    Args:
      sample_rate: Sample rate of audio in the dataset.
      frame_rate: Frame rate of features in the dataset.
    """
    self._sample_rate = sample_rate
    self._frame_rate = frame_rate

  @property
  def sample_rate(self):
    """Return dataset sample rate, must be defined in the constructor."""
    return self._sample_rate

  @property
  def frame_rate(self):
    """Return dataset feature frame rate, must be defined in the constructor."""
    return self._frame_rate

  def get_dataset(self, shuffle):
    """A method that returns a tf.data.Dataset."""
    raise NotImplementedError

  def get_batch(self,
                batch_size,
                shuffle=True,
                repeats=-1,
                drop_remainder=True):
    """Read dataset.

    Args:
      batch_size: Size of batch.
      shuffle: Whether to shuffle the examples.
      repeats: Number of times to repeat dataset. -1 for endless repeats.
      drop_remainder: Whether the last batch should be dropped.

    Returns:
      A batched tf.data.Dataset.
    """
    dataset = self.get_dataset(shuffle)
    dataset = dataset.repeat(repeats)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(buffer_size=_AUTOTUNE)
    return dataset


@gin.register
class ExperimentalDataProvider(DataProvider):
  """Use the new tf.data.experimental.save/load() interface."""

  def __init__(self, data_dir, sample_rate, frame_rate):
    """RecordProvider constructor."""
    super().__init__(sample_rate, frame_rate)
    self.data_dir = data_dir

  def get_dataset(self, shuffle=True):
    """Read dataset direct from disk.

    Args:
      shuffle: Unused.

    Returns:
      dataset: A tf.dataset that reads from new experimental format.
    """
    return tf.data.experimental.load(self.data_dir)


class TfdsProvider(DataProvider):
  """Base class for reading datasets from TensorFlow Datasets (TFDS)."""

  def __init__(self, name, split, data_dir, sample_rate, frame_rate):
    """TfdsProvider constructor.

    Args:
      name: TFDS dataset name (with optional config and version).
      split: Dataset split to use of the TFDS dataset.
      data_dir: The directory to read TFDS datasets from. Defaults to
        "~/tensorflow_datasets".
      sample_rate: Sample rate of audio in the dataset.
      frame_rate: Frame rate of features in the dataset.
    """
    self._name = name
    self._split = split
    self._data_dir = data_dir
    super().__init__(sample_rate, frame_rate)

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
               data_dir='gs://tfds-data/datasets',
               sample_rate=16000,
               frame_rate=250,
               include_note_labels=True):
    """TfdsProvider constructor.

    Args:
      name: TFDS dataset name (with optional config and version).
      split: Dataset split to use of the TFDS dataset.
      data_dir: The directory to read the prepared NSynth dataset from. Defaults
        to the public TFDS GCS bucket.
      sample_rate: Sample rate of audio in the dataset.
      frame_rate: Frame rate of features in the dataset.
      include_note_labels: Return dataset without note-level labels
        (pitch, instrument).
    """
    self._include_note_labels = include_note_labels
    if data_dir == 'gs://tfds-data/datasets':
      logging.warning(
          'Using public TFDS GCS bucket to load NSynth. If not running on '
          'GCP, this will be very slow, and it is recommended you prepare '
          'the dataset locally with TFDS and set the data_dir appropriately.')
    super().__init__(name, split, data_dir, sample_rate, frame_rate)

  def get_dataset(self, shuffle=True):
    """Returns dataset with slight restructuring of feature dictionary."""
    def preprocess_ex(ex):
      ex_out = {
          'audio':
              ex['audio'],
          'f0_hz':
              ex['f0']['hz'],
          'f0_confidence':
              ex['f0']['confidence'],
          'loudness_db':
              ex['loudness']['db'],
      }
      if self._include_note_labels:
        ex_out.update({
            'pitch':
                ex['pitch'],
            'instrument_source':
                ex['instrument']['source'],
            'instrument_family':
                ex['instrument']['family'],
            'instrument':
                ex['instrument']['label'],
        })
      return ex_out

    dataset = super().get_dataset(shuffle)
    dataset = dataset.map(preprocess_ex, num_parallel_calls=_AUTOTUNE)
    return dataset


@gin.register
class TFRecordProvider(DataProvider):
  """Class for reading TFRecords and returning a dataset."""

  def __init__(self,
               file_pattern=None,
               example_secs=4,
               sample_rate=16000,
               frame_rate=250,
               centered=False):
    """RecordProvider constructor."""
    super().__init__(sample_rate, frame_rate)
    self._file_pattern = file_pattern or self.default_file_pattern
    self._audio_length = example_secs * sample_rate
    self._audio_16k_length = example_secs * CREPE_SAMPLE_RATE
    self._feature_length = self.get_feature_length(centered)

  def get_feature_length(self, centered):
    """Take into account center padding to get number of frames."""
    # Number of frames is independent of frame size for "center/same" padding.
    hop_size = CREPE_SAMPLE_RATE / self.frame_rate
    padding = 'center' if centered else 'same'
    return get_framed_lengths(
        self._audio_16k_length, CREPE_FRAME_SIZE, hop_size, padding)[0]

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
        'audio_16k':
            tf.io.FixedLenFeature([self._audio_16k_length], dtype=tf.float32),
        'f0_hz':
            tf.io.FixedLenFeature([self._feature_length], dtype=tf.float32),
        'f0_confidence':
            tf.io.FixedLenFeature([self._feature_length], dtype=tf.float32),
        'loudness_db':
            tf.io.FixedLenFeature([self._feature_length], dtype=tf.float32),
    }


@gin.register
class LegacyTFRecordProvider(TFRecordProvider):
  """Class for reading TFRecords and returning a dataset."""

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


# ------------------------------------------------------------------------------
# Multi-dataset DataProviders
# ------------------------------------------------------------------------------
@gin.register
class BaseMultiProvider(DataProvider):
  """Base class for providers that combine multiple datasets."""

  def __init__(self, data_providers, batch_size_ratios=()):
    """Constructor.

    Args:
      data_providers: A list of data_providers.
      batch_size_ratios: A list of ratios of batch sizes for each provider.
        These do not need to sum to 1. For example, [2, 1] will produce batches
        with a size ratio of 2 to 1.
    """
    if batch_size_ratios:
      # Check lengths match.
      if len(batch_size_ratios) != len(data_providers):
        raise ValueError('List of batch size ratios ({}) must be of the same '
                         'length as the list of data providers ({}) for varying'
                         'batch sizes.'.format(
                             len(batch_size_ratios), len(data_providers)))
      # Normalize the ratios.
      total = sum(batch_size_ratios)
      batch_size_ratios = [float(bsr) / total for bsr in batch_size_ratios]
    else:
      # Sample evenly from each.
      batch_size_ratios = [1.0 for _ in data_providers]

    # Make sure all sample rates are the same.
    sample_rates = [dp.sample_rate for dp in data_providers]
    assert len(set(sample_rates)) <= 1
    sample_rate = sample_rates[0]

    # Make sure all frame rates are the same.
    frame_rates = [dp.frame_rate for dp in data_providers]
    assert len(set(frame_rates)) <= 1
    frame_rate = frame_rates[0]

    super().__init__(sample_rate, frame_rate)
    self._data_providers = data_providers
    self._batch_size_ratios = batch_size_ratios


@gin.register
class ZippedProvider(BaseMultiProvider):
  """Combines datasets from two providers with zip."""

  def get_dataset(self, shuffle=True):
    """Read dataset.

    Args:
      shuffle: Whether to shuffle the input files.

    Returns:
      dataset: A zipped tf.data.Dataset from multiple providers.
    """
    datasets = tuple(dp.get_dataset(shuffle) for dp in self._data_providers)
    return tf.data.Dataset.zip(datasets)

  def get_batch(self,
                batch_size,
                shuffle=True,
                repeats=-1,
                drop_remainder=False):
    """Read dataset.

    Args:
      batch_size: Size of batches, can be a list to have varying batch_sizes.
      shuffle: Whether to shuffle the examples.
      repeats: Number of times to repeat dataset. -1 for endless repeats.
      drop_remainder: Whether the last batch should be dropped.

    Returns:
      A batched tf.data.Dataset.
    """
    if not self._batch_size_ratios:
      # One batch size for all datasets ('None' is batch shape).
      return super().get_batch(batch_size)

    else:
      # Varying batch sizes (Integer batch shape for each).
      batch_sizes = [int(batch_size * bsr) for bsr in self._batch_size_ratios]
      datasets = tuple(
          dp.get_dataset(shuffle).batch(bs, drop_remainder=drop_remainder)
          for bs, dp in zip(batch_sizes, self._data_providers))
      dataset = tf.data.Dataset.zip(datasets)
      dataset = dataset.repeat(repeats)
      dataset = dataset.prefetch(buffer_size=_AUTOTUNE)
      return dataset


@gin.register
class MixedProvider(BaseMultiProvider):
  """Combines datasets from two providers mixed with sampling."""

  def get_dataset(self, shuffle=True):
    """Read dataset.

    Args:
      shuffle: Whether to shuffle the input files.

    Returns:
      dataset: A tf.data.Dataset mixed from multiple datasets.
    """
    datasets = tuple(dp.get_dataset(shuffle) for dp in self._data_providers)
    return tf.data.experimental.sample_from_datasets(
        datasets, self._batch_size_ratios)


# ------------------------------------------------------------------------------
# Synthetic Data for InverseSynthesis
# ------------------------------------------------------------------------------
@gin.register
class SyntheticNotes(LegacyTFRecordProvider):
  """Create self-supervised control signal.

  EXPERIMENTAL

  Pass file_pattern to tfrecords created by `ddsp_generate_synthetic_data.py`.
  """

  def __init__(self,
               n_timesteps,
               n_harmonics,
               n_mags,
               file_pattern=None,
               sample_rate=16000):
    self.n_timesteps = n_timesteps
    self.n_harmonics = n_harmonics
    self.n_mags = n_mags
    super().__init__(file_pattern=file_pattern, sample_rate=sample_rate)

  @property
  def features_dict(self):
    """Dictionary of features to read from dataset."""
    return {
        'f0_hz':
            tf.io.FixedLenFeature([self.n_timesteps, 1], dtype=tf.float32),
        'harm_amp':
            tf.io.FixedLenFeature([self.n_timesteps, 1], dtype=tf.float32),
        'harm_dist':
            tf.io.FixedLenFeature(
                [self.n_timesteps, self.n_harmonics], dtype=tf.float32),
        'sin_amps':
            tf.io.FixedLenFeature(
                [self.n_timesteps, self.n_harmonics], dtype=tf.float32),
        'sin_freqs':
            tf.io.FixedLenFeature(
                [self.n_timesteps, self.n_harmonics], dtype=tf.float32),
        'noise_magnitudes':
            tf.io.FixedLenFeature(
                [self.n_timesteps, self.n_mags], dtype=tf.float32),
    }


@gin.register
class Urmp(LegacyTFRecordProvider):
  """Urmp training set."""

  def __init__(self,
               base_dir,
               instrument_key='tpt',
               split='train',
               suffix=None):
    """URMP dataset for either a specific instrument or all instruments.

    Args:
      base_dir: Base directory to URMP TFRecords.
      instrument_key: Determines which instrument to return. Choices include
        ['all', 'bn', 'cl', 'db', 'fl', 'hn', 'ob', 'sax', 'tba', 'tbn',
        'tpt', 'va', 'vc', 'vn'].
      split: Choices include ['train', 'test'].
      suffix: Choices include [None, 'batched', 'unbatched'], but broadly
        applies to any suffix adding to the file pattern.
        When suffix is not None, will add "_{suffix}" to the file pattern.
        This option is used in gs://magentadata/datasets/urmp/urmp_20210324.
        With the "batched" suffix, the dataloader will load tfrecords
        containing segmented audio samples in 4 seconds. With the "unbatched"
        suffix, the dataloader will load tfrecords containing unsegmented
        samples which could be used for learning note sequence in URMP dataset.

    """
    self.instrument_key = instrument_key
    self.split = split
    self.base_dir = base_dir
    self.suffix = '' if suffix is None else '_' + suffix
    super().__init__()

  @property
  def default_file_pattern(self):
    if self.instrument_key == 'all':
      file_pattern = 'all_instruments_{}{}.tfrecord*'.format(
          self.split, self.suffix)
    else:
      file_pattern = 'urmp_{}_solo_ddsp_conditioning_{}{}.tfrecord*'.format(
          self.instrument_key, self.split, self.suffix)

    return os.path.join(self.base_dir, file_pattern)


@gin.register
class UrmpMidi(Urmp):
  """Urmp training set with midi note data.

  This class loads the segmented data in tfrecord that contains 4-second audio
  clips of the URMP dataset. To load tfrecord that contains unsegmented full
  piece of URMP recording, please use `UrmpMidiUnsegmented` class instead.
  """

  _INSTRUMENTS = ['vn', 'va', 'vc', 'db', 'fl', 'ob', 'cl', 'sax', 'bn', 'tpt',
                  'hn', 'tbn', 'tba']

  @property
  def features_dict(self):
    base_features = super().features_dict
    base_features.update({
        'note_active_velocities':
            tf.io.FixedLenFeature([self._feature_length * 128], tf.float32),
        'note_active_frame_indices':
            tf.io.FixedLenFeature([self._feature_length * 128], tf.float32),
        'instrument_id': tf.io.FixedLenFeature([], tf.string),
        'recording_id': tf.io.FixedLenFeature([], tf.string),
        'power_db':
            tf.io.FixedLenFeature([self._feature_length], dtype=tf.float32),
        'note_onsets':
            tf.io.FixedLenFeature([self._feature_length * 128],
                                  dtype=tf.float32),
        'note_offsets':
            tf.io.FixedLenFeature([self._feature_length * 128],
                                  dtype=tf.float32),
    })
    return base_features

  def get_dataset(self, shuffle=True):

    instrument_ids = range(len(self._INSTRUMENTS))
    inst_vocab = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(self._INSTRUMENTS, instrument_ids),
        -1)

    def _reshape_tensors(data):
      data['note_active_frame_indices'] = tf.reshape(
          data['note_active_frame_indices'], (-1, 128))
      data['note_active_velocities'] = tf.reshape(
          data['note_active_velocities'], (-1, 128))
      data['instrument_id'] = inst_vocab.lookup(data['instrument_id'])
      data['midi'] = tf.argmax(data['note_active_frame_indices'], axis=-1)
      data['f0_hz'] = data['f0_hz'][..., tf.newaxis]
      data['loudness_db'] = data['loudness_db'][..., tf.newaxis]
      onsets = tf.reduce_sum(
          tf.reshape(data['note_onsets'], (-1, 128)), axis=-1)
      data['onsets'] = tf.cast(onsets > 0, tf.int64)
      offsets = tf.reduce_sum(
          tf.reshape(data['note_offsets'], (-1, 128)), axis=-1)
      data['offsets'] = tf.cast(offsets > 0, tf.int64)

      return data

    ds = super().get_dataset(shuffle)
    ds = ds.map(_reshape_tensors, num_parallel_calls=_AUTOTUNE)
    return ds


class UrmpMidiUnsegmented(Urmp):
  """Urmp dataset using unsegmented data.

  Unsegmented here means that the data samples are not segmented to 4-second
  chunks as in UrmpMidi dataset.
  """

  _INSTRUMENTS = ['vn', 'va', 'vc', 'db', 'fl', 'ob', 'cl', 'sax', 'bn', 'tpt',
                  'hn', 'tbn', 'tba']

  @property
  def features_dict(self):
    base_features = {
        'audio':
            tf.io.VarLenFeature(dtype=tf.float32),
        'f0_hz':
            tf.io.VarLenFeature(dtype=tf.float32),
        'f0_confidence':
            tf.io.VarLenFeature(dtype=tf.float32),
        'loudness_db':
            tf.io.VarLenFeature(dtype=tf.float32),
    }
    base_features.update({
        'note_active_velocities':
            tf.io.VarLenFeature(tf.float32),
        'note_active_frame_indices':
            tf.io.VarLenFeature(tf.float32),
        'instrument_id': tf.io.FixedLenFeature([], tf.string),
        'recording_id': tf.io.FixedLenFeature([], tf.string),
        'power_db':
            tf.io.VarLenFeature(dtype=tf.float32),
        'note_onsets':
            tf.io.VarLenFeature(dtype=tf.float32),
        'note_offsets':
            tf.io.VarLenFeature(dtype=tf.float32),
    })
    return base_features

  def get_dataset(self, shuffle=True):
    instrument_ids = range(len(self._INSTRUMENTS))
    inst_vocab = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            self._INSTRUMENTS, instrument_ids), -1)

    def _reshape_tensors(data):
      data['audio'] = tf.sparse.to_dense(data['audio'])
      data['note_active_frame_indices'] = tf.reshape(
          tf.sparse.to_dense(data['note_active_frame_indices']), (-1, 128))
      data['note_active_velocities'] = tf.reshape(
          tf.sparse.to_dense(data['note_active_velocities']), (-1, 128))
      data['instrument_id'] = inst_vocab.lookup(data['instrument_id'])
      data['midi'] = tf.argmax(data['note_active_frame_indices'], axis=-1)
      data['f0_hz'] = tf.sparse.to_dense(data['f0_hz'])[..., tf.newaxis]
      data['loudness_db'] = tf.sparse.to_dense(data['loudness_db'])[
          ..., tf.newaxis]
      # reshape and rename things
      onsets = tf.reduce_sum(
          tf.reshape(tf.sparse.to_dense(data['note_onsets']), (-1, 128)),
          axis=-1)
      data['onsets'] = tf.cast(onsets > 0, tf.int64)
      offsets = tf.reduce_sum(
          tf.reshape(tf.sparse.to_dense(data['note_offsets']), (-1, 128)),
          axis=-1)
      data['offsets'] = tf.cast(offsets > 0, tf.int64)

      return data

    ds = super().get_dataset(shuffle)
    ds = ds.map(_reshape_tensors, num_parallel_calls=_AUTOTUNE)
    return ds


