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

"""Apache Beam pipeline for computing TFRecord dataset from audio files."""

from absl import logging
import apache_beam as beam
from ddsp import spectral_ops
import numpy as np
import pydub
import tensorflow.compat.v2 as tf

CREPE_SAMPLE_RATE = spectral_ops.CREPE_SAMPLE_RATE  # 16kHz.


def _load_audio_as_array(audio_path, sample_rate):
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
    sample_arr = audio_segment.get_array_of_samples()
    audio = np.array(sample_arr).astype(np.float32)
    # Zero pad missing samples, if any
    audio = spectral_ops.pad_or_trim_to_expected_length(audio, expected_len)
  # Convert from int to float representation.
  audio /= np.iinfo(sample_arr.typecode).max
  return audio


def _load_audio(audio_path, sample_rate):
  """Load audio file."""
  logging.info("Loading '%s'.", audio_path)
  beam.metrics.Metrics.counter('prepare-tfrecord', 'load-audio').inc()
  audio = _load_audio_as_array(audio_path, sample_rate)
  if sample_rate != CREPE_SAMPLE_RATE:
    audio_16k = _load_audio_as_array(audio_path, CREPE_SAMPLE_RATE)
  else:
    audio_16k = audio
  return {'audio': audio, 'audio_16k': audio_16k}


def _chunk_audio(ex, sample_rate, chunk_secs):
  """Pad audio and split into chunks."""
  beam.metrics.Metrics.counter('prepare-tfrecord', 'load-audio').inc()
  def get_chunks(audio, sample_rate):
    chunk_size = int(chunk_secs * sample_rate)
    return tf.signal.frame(audio, chunk_size, chunk_size, pad_end=True).numpy()

  chunks = get_chunks(ex['audio'], sample_rate)
  chunks_16k = get_chunks(ex['audio_16k'], CREPE_SAMPLE_RATE)
  assert chunks.shape[0] == chunks_16k.shape[0]
  n_chunks = chunks.shape[0]
  for i in range(n_chunks):
    yield {'audio': chunks[i], 'audio_16k': chunks_16k[i]}


def _add_f0_estimate(ex, frame_rate, center, viterbi):
  """Add fundamental frequency (f0) estimate using CREPE."""
  beam.metrics.Metrics.counter('prepare-tfrecord', 'estimate-f0').inc()
  audio = ex['audio_16k']
  padding = 'center' if center else 'same'
  f0_hz, f0_confidence = spectral_ops.compute_f0(
      audio, frame_rate, viterbi=viterbi, padding=padding)
  ex = dict(ex)
  ex.update({
      'f0_hz': f0_hz.astype(np.float32),
      'f0_confidence': f0_confidence.astype(np.float32)
  })
  return ex


def _add_loudness(ex, frame_rate, n_fft, center):
  """Add loudness in dB."""
  beam.metrics.Metrics.counter('prepare-tfrecord', 'compute-loudness').inc()
  audio = ex['audio_16k']
  padding = 'center' if center else 'same'
  loudness_db = spectral_ops.compute_loudness(
      audio, CREPE_SAMPLE_RATE, frame_rate, n_fft, padding=padding)
  ex = dict(ex)
  ex['loudness_db'] = loudness_db.numpy().astype(np.float32)
  return ex


def _split_example(ex, sample_rate, frame_rate, example_secs, hop_secs, center):
  """Splits example into windows, padding final window if needed."""

  def get_windows(sequence, rate, center):
    window_size = int(example_secs * rate)
    if center:
      window_size += 1
    hop_size = int(hop_secs * rate)
    # Don't pad the end.
    n_windows = int(np.floor((len(sequence) - window_size) / hop_size)) + 1
    for i in range(n_windows):
      start = i * hop_size
      end = start + window_size
      yield sequence[start:end]

  for audio, audio_16k, loudness_db, f0_hz, f0_confidence in zip(
      get_windows(ex['audio'], sample_rate, center=False),
      get_windows(ex['audio_16k'], CREPE_SAMPLE_RATE, center=False),
      get_windows(ex['loudness_db'], frame_rate, center),
      get_windows(ex['f0_hz'], frame_rate, center),
      get_windows(ex['f0_confidence'], frame_rate, center)):
    beam.metrics.Metrics.counter('prepare-tfrecord', 'split-example').inc()
    yield {
        'audio': audio,
        'audio_16k': audio_16k,
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
          }))


def _add_key(example):
  """Add a key to this example by taking the hash of the values."""
  return hash(example['audio'].tobytes()), example


def _eval_split_partition_fn(example, num_partitions, eval_fraction, all_ids):
  """Partition function to split into train/eval based on the hash ids."""
  del num_partitions
  example_id = example[0]
  eval_range = int(len(all_ids) * eval_fraction)
  for i in range(eval_range):
    if all_ids[i] == example_id:
      return 0
  return 1


def prepare_tfrecord(input_audio_paths,
                     output_tfrecord_path,
                     num_shards=None,
                     sample_rate=16000,
                     frame_rate=250,
                     example_secs=4,
                     hop_secs=1,
                     eval_split_fraction=0.0,
                     chunk_secs=20.0,
                     center=False,
                     viterbi=True,
                     pipeline_options=()):
  """Prepares a TFRecord for use in training, evaluation, and prediction.

  Args:
    input_audio_paths: An iterable of paths to audio files to include in
      TFRecord.
    output_tfrecord_path: The prefix path to the output TFRecord. Shard numbers
      will be added to actual path(s).
    num_shards: The number of shards to use for the TFRecord. If None, this
      number will be determined automatically.
    sample_rate: The sample rate to use for the audio.
    frame_rate: The frame rate to use for f0 and loudness features. If set to
      None, these features will not be computed.
    example_secs: The size of the sliding window (in seconds) to use to split
      the audio and features. If 0, they will not be split.
    hop_secs: The number of seconds to hop when computing the sliding windows.
    eval_split_fraction: Fraction of the dataset to reserve for eval split. If
      set to 0, no eval split is created.
    chunk_secs: Chunk size in seconds used to split the input audio
      files. This is used to split large audio files into manageable chunks for
      better parallelization and to enable non-overlapping train/eval splits.
    center: Provide zero-padding to audio so that frame timestamps will be
      centered.
    viterbi: Use viterbi decoding of pitch.
    pipeline_options: An iterable of command line arguments to be used as
      options for the Beam Pipeline.
  """
  def postprocess_pipeline(examples, output_path, stage_name=''):
    """After chunking, features, and train-eval split, create TFExamples."""
    if stage_name:
      stage_name = f'_{stage_name}'

    if example_secs:
      examples |= f'split_examples{stage_name}' >> beam.FlatMap(
          _split_example,
          sample_rate=sample_rate,
          frame_rate=frame_rate,
          example_secs=example_secs,
          hop_secs=hop_secs,
          center=center)
    _ = (
        examples
        | f'reshuffle{stage_name}' >> beam.Reshuffle()
        | f'make_tfexample{stage_name}' >> beam.Map(_float_dict_to_tfexample)
        | f'write{stage_name}' >> beam.io.tfrecordio.WriteToTFRecord(
            output_path,
            num_shards=num_shards,
            coder=beam.coders.ProtoCoder(tf.train.Example)))

  # Start the pipeline.
  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      pipeline_options)
  with beam.Pipeline(options=pipeline_options) as pipeline:
    examples = (
        pipeline
        | beam.Create(input_audio_paths)
        | beam.Map(_load_audio, sample_rate))

    # Split into chunks for train/eval split and better parallelism.
    if chunk_secs:
      examples |= beam.FlatMap(
          _chunk_audio,
          sample_rate=sample_rate,
          chunk_secs=chunk_secs)

    # Add features.
    if frame_rate:
      examples = (
          examples
          | beam.Map(_add_f0_estimate,
                     frame_rate=frame_rate,
                     center=center,
                     viterbi=viterbi)
          | beam.Map(_add_loudness,
                     frame_rate=frame_rate,
                     n_fft=512,
                     center=center))

    # Create train/eval split.
    if eval_split_fraction:
      examples |= beam.Map(_add_key)
      keys = examples | beam.Keys()
      splits = examples | beam.Partition(_eval_split_partition_fn, 2,
                                         eval_split_fraction,
                                         beam.pvalue.AsList(keys))

      # Remove ids.
      eval_split = splits[0] | 'remove_id_eval' >> beam.Map(lambda x: x[1])
      train_split = splits[1] | 'remove_id_train' >> beam.Map(lambda x: x[1])

      postprocess_pipeline(eval_split, f'{output_tfrecord_path}-eval', 'eval')
      postprocess_pipeline(train_split, f'{output_tfrecord_path}-train',
                           'train')
    else:
      postprocess_pipeline(examples, output_tfrecord_path)
