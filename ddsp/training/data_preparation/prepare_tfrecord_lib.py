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
"""Apache Beam pipeline for computing TFRecord dataset from audio files."""

from absl import logging
import apache_beam as beam
from ddsp.spectral_ops import _CREPE_SAMPLE_RATE
from ddsp.spectral_ops import compute_f0
from ddsp.spectral_ops import compute_loudness
from ddsp.spectral_ops import LD_RANGE
import numpy as np
import pydub
import tensorflow.compat.v2 as tf



def _load_audio_as_array(audio_path: str,
                         sample_rate: int) -> np.array:
  """Load audio file at specified sample rate and return an array.

  When `sample_rate` > original SR of audio file, Pydub may miss samples when
  reading file defined in `audio_path`. Must manually zero-pad missing samples.

  Args:
    audio_path: path to audio file
    sample_rate: desired sample rate (can be different from original SR)

  Returns:
    audio: audio in np.float32
  """
  with tf.io.gfile.GFile(audio_path, 'rb') as f:
    # Load audio at original SR
    audio_segment = (pydub.AudioSegment.from_file(f).set_channels(1))
    # Compute expected length at given `sample_rate`
    expected_len = int(audio_segment.duration_seconds * sample_rate)
    # Resample to `sample_rate`
    audio_segment = audio_segment.set_frame_rate(sample_rate)
    audio = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    # Zero pad missing samples, if any
    audio = _make_array_expected_length(audio, expected_len)
  # Convert from int to float representation.
  audio /= 2**(8 * audio_segment.sample_width)
  return audio


def _make_array_expected_length(array, expected_len, pad_value=0):
  """Make array equal to the expected length."""
  if len(array) < expected_len:
    # Pad missing samples
    n_padding = expected_len - len(array)
    array = np.concatenate([array, np.ones(n_padding) * pad_value], axis=0)
  elif len(array) > expected_len:
    # Trim extra samples
    array = array[:expected_len]
  return array


def _load_audio(audio_path, sample_rate):
  """Load audio file."""
  logging.info("Loading '%s'.", audio_path)
  beam.metrics.Metrics.counter('prepare-tfrecord', 'load-audio').inc()
  audio = _load_audio_as_array(audio_path, sample_rate)
  # Crepe pitch extraction only works at 16Khz sample rate
  audio_crepe = _load_audio_as_array(audio_path, _CREPE_SAMPLE_RATE)
  return {'audio': audio, 'audio_crepe': audio_crepe}


def _add_loudness(ex, sample_rate, frame_rate, n_fft=2048):
  """Add loudness in dB."""
  beam.metrics.Metrics.counter('prepare-tfrecord', 'compute-loudness').inc()
  audio = ex['audio']
  expected_len = int(len(audio) / sample_rate * frame_rate)
  mean_loudness_db = compute_loudness(audio, sample_rate, frame_rate, n_fft)
  # Trim `mean_loudness_db` or pad to dB floor
  mean_loudness_db = _make_array_expected_length(mean_loudness_db, expected_len,
                                                 -LD_RANGE)
  ex = dict(ex)
  ex['loudness_db'] = mean_loudness_db.astype(np.float32)
  return ex


def _add_f0_estimate(ex, frame_rate):
  """Add fundamental frequency (f0) estimate using CREPE."""
  beam.metrics.Metrics.counter('prepare-tfrecord', 'estimate-f0').inc()
  audio = ex['audio_crepe']
  f0_hz, f0_confidence = compute_f0(audio, _CREPE_SAMPLE_RATE, frame_rate)
  ex = dict(ex)
  ex.update({
      'f0_hz': f0_hz.astype(np.float32),
      'f0_confidence': f0_confidence.astype(np.float32)
  })
  return ex


def _split_example(
    ex, sample_rate, frame_rate, window_secs, hop_secs):
  """Splits example into windows, padding final window if needed."""

  def get_windows(sequence, rate):
    window_size = int(window_secs * rate)
    hop_size = int(hop_secs * rate)
    n_windows = int(np.ceil((len(sequence) - window_size) / hop_size))  + 1
    n_samples_padded = (n_windows - 1) * hop_size + window_size
    n_padding = n_samples_padded - len(sequence)
    sequence = np.pad(sequence, (0, n_padding), mode='constant')
    for window_end in range(window_size, len(sequence) + 1, hop_size):
      yield sequence[window_end-window_size:window_end]

  for audio, audio_crepe, loudness_db, f0_hz, f0_confidence in zip(
      get_windows(ex['audio'], sample_rate),
      get_windows(ex['audio_crepe'], _CREPE_SAMPLE_RATE),
      get_windows(ex['loudness_db'], frame_rate),
      get_windows(ex['f0_hz'], frame_rate),
      get_windows(ex['f0_confidence'], frame_rate)):
    beam.metrics.Metrics.counter('prepare-tfrecord', 'split-example').inc()
    yield {
        'audio': audio,
        'audio_crepe': audio_crepe,
        'loudness_db': loudness_db,
        'f0_hz': f0_hz,
        'f0_confidence': f0_confidence
    }


def _float_dict_to_tfexample(float_dict):
  """Convert dictionary of float arrays to tf.train.Example proto."""
  return tf.train.Example(
      features=tf.train.Features(
          feature={
              k: tf.train.Feature(float_list=tf.train.FloatList(value=v))
              for k, v in float_dict.items()
          }
      ))


def prepare_tfrecord(
    input_audio_paths,
    output_tfrecord_path,
    num_shards=None,
    sample_rate=16000,
    frame_rate=250,
    window_secs=4,
    hop_secs=1,
    pipeline_options=''):
  """Prepares a TFRecord for use in training, evaluation, and prediction.

  Args:
    input_audio_paths: An iterable of paths to audio files to include in
      TFRecord.
    output_tfrecord_path: The prefix path to the output TFRecord. Shard numbers
      will be added to actual path(s).
    num_shards: The number of shards to use for the TFRecord. If None, this
      number will be determined automatically.
    sample_rate: The sample rate to use for the audio.
    frame_rate: The frame rate to use for f0 and loudness features.
      If set to None, these features will not be computed.
    window_secs: The size of the sliding window (in seconds) to use to
      split the audio and features. If 0, they will not be split.
    hop_secs: The number of seconds to hop when computing the sliding
      windows.
    pipeline_options: An iterable of command line arguments to be used as
      options for the Beam Pipeline.
  """
  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      pipeline_options)
  with beam.Pipeline(options=pipeline_options) as pipeline:
    examples = (
        pipeline
        | beam.Create(input_audio_paths)
        | beam.Map(_load_audio, sample_rate))

    if frame_rate:
      examples = (
          examples
          | beam.Map(_add_f0_estimate, frame_rate)
          | beam.Map(_add_loudness, sample_rate, frame_rate))

    if window_secs:
      examples |= beam.FlatMap(
          _split_example, sample_rate, frame_rate, window_secs, hop_secs)

    _ = (
        examples
        | beam.Reshuffle()
        | beam.Map(_float_dict_to_tfexample)
        | beam.io.tfrecordio.WriteToTFRecord(
            output_tfrecord_path,
            num_shards=num_shards,
            coder=beam.coders.ProtoCoder(tf.train.Example))
    )
