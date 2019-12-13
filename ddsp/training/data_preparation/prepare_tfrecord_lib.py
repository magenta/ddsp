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
"""Apache Beam pipeline for computing TFRecord dataset from audio files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

from absl import logging
import apache_beam as beam
import crepe
from ddsp.spectral_ops import calc_loudness
import librosa
import numpy as np
import tensorflow.compat.v1 as tf


_CREPE_SAMPLE_RATE = 16000
_CREPE_FRAME_SIZE = 1024


def _load_and_split_audio(
    audio_path, audio_rate, window_size, hop_size, pad_end=True):
  """Load and split audio file with sliding window."""
  logging.info("Loading and splitting '%s'.", audio_path)
  beam.metrics.Metrics.counter('prepare-tfrecord', 'load-audio').inc()
  with tempfile.NamedTemporaryFile(suffix='.wav') as f:
    tf.io.gfile.copy(audio_path, f.name, overwrite=True)
    audio, _ = librosa.load(f.name, sr=audio_rate)

  window_size = window_size or len(audio)
  if pad_end:
    n_windows = int(np.ceil((len(audio) - window_size) / hop_size))  + 1
    n_samples_padded = (n_windows - 1) * hop_size + window_size
    n_padding = n_samples_padded - len(audio)
    audio = np.pad(audio, (0, n_padding), mode='constant')
  for window_end in range(window_size, len(audio) + 1, hop_size):
    beam.metrics.Metrics.counter('prepare-tfrecord', 'split-audio').inc()
    yield {'audio': audio[window_end-window_size:window_end]}


def _add_loudness(ex, audio_rate, loudness_rate):
  """Add uncenetered loudness."""
  beam.metrics.Metrics.counter('prepare-tfrecord', 'compute-loudness').inc()
  with tf.Session() as sess:
    mean_loudness_db = sess.run(
        calc_loudness(ex['audio'], hop_size=audio_rate//loudness_rate)[0])

  ex = dict(ex)
  ex['loudness_uncentered'] = mean_loudness_db.astype(np.float32)
  return ex


def _add_f0_estimate(ex, audio_rate, f0_rate):
  """Add fundamental frequency (f0) estimate using CREPE."""
  beam.metrics.Metrics.counter('prepare-tfrecord', 'estimate-f0').inc()
  audio = ex['audio']

  # Pad end so that `n_frames = n_sec * f0_rate`.
  # We compute this using the CREPE model's sample rate and then convert back to
  # our sample rate.
  n_secs = len(audio) / audio_rate
  n_samples = n_secs * _CREPE_SAMPLE_RATE
  hop_size = audio_rate / f0_rate
  frame_size = _CREPE_FRAME_SIZE
  n_frames = np.ceil(n_secs * f0_rate)
  n_samples_padded = (n_frames - 1) * hop_size + frame_size
  n_padding = (n_samples_padded - n_samples) * audio_rate / _CREPE_SAMPLE_RATE
  assert n_padding % 1 == 0
  audio = np.pad(audio, (0, int(n_padding)), mode='constant')

  _, f0_hz, f0_confidence, _ = crepe.predict(
      audio,
      sr=audio_rate,
      viterbi=True,
      step_size=1000/f0_rate,
      center=False,
      verbose=0)
  f0_midi = librosa.core.hz_to_midi(f0_hz)
  # Set -infs introduced by hz_to_midi to 0.
  f0_midi[f0_midi == -np.inf] = 0
  # Set nans to 0 in confidence.
  f0_confidence = np.nan_to_num(f0_confidence)
  ex = dict(ex)
  ex.update({
      'f0': f0_hz.astype(np.float32),
      'f0_confidence': f0_confidence.astype(np.float32)
  })
  return ex


def _add_centered_loudness(features, ld_mean, ld_variance):
  """Compute centered loudness feature."""
  features = dict(features)
  features['loudness'] = (
      features['loudness_uncentered'] - ld_mean / np.sqrt(ld_variance))
  return features


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
    audio_rate=16000,
    f0_and_loudness_rate=250,
    window_size=16000*4,
    hop_size=16000,
    pipeline_options=''):
  """Prepares a TFRecord for use in training, evaluation, and prediction.

  Args:
    input_audio_paths: An iterable of paths to audio files to include in
      TFRecord.
    output_tfrecord_path: The prefix path to the output TFRecord. Shard numbers
      will be added to actual path(s).
    num_shards: The number of shards to use for the TFRecord. If None, this
      number will be determined automatically.
    audio_rate: The sample rate to use for the audio.
    f0_and_loudness_rate: The sample rate to use for f0 and loudness features.
      If set to None, these features will not be computed.
    window_size: The size of the sliding window (in audio samples) to use to
      split the input audio. If None, the audio will not be split.
    hop_size: The number of audio samples to hop when computing the sliding
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
        | beam.FlatMap(_load_and_split_audio,
                       audio_rate=audio_rate,
                       window_size=window_size,
                       hop_size=hop_size))
    if f0_and_loudness_rate:
      examples = (
          examples
          | beam.Map(_add_f0_estimate, audio_rate, f0_and_loudness_rate)
          | beam.Map(_add_loudness, audio_rate, f0_and_loudness_rate))

      # Compute centered loudness.
      loudness = (
          examples
          | beam.FlatMap(lambda x: x['loudness_uncentered']))
      loudness_mean = (
          loudness
          | 'loudness_mean' >> beam.combiners.Mean.Globally())
      loudness_variance = (
          loudness
          | beam.Map(lambda ld, ld_mean: (ld - ld_mean)**2,
                     ld_mean=beam.pvalue.AsSingleton(loudness_mean))
          | 'loudness_variance' >> beam.combiners.Mean.Globally())
      examples = (
          examples
          | beam.Map(_add_centered_loudness,
                     ld_mean=beam.pvalue.AsSingleton(loudness_mean),
                     ld_variance=beam.pvalue.AsSingleton(loudness_variance)))
    _ = (
        examples
        | beam.Reshuffle()
        | beam.Map(_float_dict_to_tfexample)
        | beam.io.tfrecordio.WriteToTFRecord(
            output_tfrecord_path,
            num_shards=num_shards,
            coder=beam.coders.ProtoCoder(tf.train.Example))
    )

