# Copyright 2022 The DDSP Authors.
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

"""URMP data import pipeline."""
import apache_beam as beam
import ddsp
from ddsp.training import heuristics
from mir_eval import melody
from note_seq import audio_io
from note_seq import constants
from note_seq import sequences_lib
from note_seq.protobuf import music_pb2
import numpy as np
import tensorflow as tf


DDSP_SAMPLE_RATE = 250
AUDIO_SAMPLE_RATE = 16000


def parse_example(tfexample):
  """Parse tf.Example protos to dict of numpy arrays."""
  features = {
      'id':
          tf.io.FixedLenFeature([], dtype=tf.string),
      'audio':
          tf.io.FixedLenFeature([], dtype=tf.string),
      'f0_hz':
          tf.io.FixedLenSequenceFeature([],
                                        dtype=tf.float32,
                                        allow_missing=True),
      'f0_time':
          tf.io.FixedLenSequenceFeature([],
                                        dtype=tf.float32,
                                        allow_missing=True),
      'sequence':
          tf.io.FixedLenFeature([], dtype=tf.string)
  }
  ex = {
      key: val.numpy()
      for key, val in tf.io.parse_single_example(tfexample, features).items()
  }
  return ex


def get_active_frame_indices(piano_roll):
  """Create matrix of frame indices for active notes relative to onset."""
  active_frame_indices = np.zeros_like(piano_roll.active_velocities)
  for frame_i in range(1, active_frame_indices.shape[0]):
    prev_indices = active_frame_indices[frame_i - 1, :]
    active_notes = piano_roll.active[frame_i, :]
    active_frame_indices[frame_i, :] = (prev_indices + 1) * active_notes
  return active_frame_indices


def attach_metadata(ex, ddsp_sample_rate, audio_sample_rate, force_monophonic):
  """Parse and attach metadata from the dataset."""

  def extract_recording_id(id_string):
    id_string = id_string.split(b'/')[-1]
    id_string = id_string.split(b'.')[0]
    return id_string

  def extract_instrument_id(id_string):
    id_string = extract_recording_id(id_string).split(b'_')
    return id_string[2]

  def extract_notes(sequence_str, expected_seconds):
    ns = music_pb2.NoteSequence.FromString(sequence_str)
    # total time in dataset doesn't include silence at the end
    if force_monophonic:
      for i in range(1, len(ns.notes)):
        note = ns.notes[i]
        prev_note = ns.notes[i - 1]
        onset_frame = int(note.start_time * ddsp_sample_rate)
        prev_note_offset_frame = int(prev_note.end_time * ddsp_sample_rate)
        if prev_note_offset_frame >= onset_frame:
          frames_to_move = (prev_note_offset_frame - onset_frame) + 1
          # move previous note's onset back by frames_to_move frames in seconds
          prev_note.end_time -= float(frames_to_move) / ddsp_sample_rate

    ns.total_time = expected_seconds
    piano_roll = sequences_lib.sequence_to_pianoroll(
        ns,
        frames_per_second=ddsp_sample_rate,
        min_pitch=constants.MIN_MIDI_PITCH,
        max_pitch=constants.MAX_MIDI_PITCH,
        onset_mode='length_ms')

    note_dict = {
        'note_active_velocities': piano_roll.active_velocities,
        'note_active_frame_indices': get_active_frame_indices(piano_roll),
        'note_onsets': piano_roll.onsets,
        'note_offsets': piano_roll.offsets
    }

    return note_dict

  ex['recording_id'] = extract_recording_id(ex['id'])
  ex['instrument_id'] = extract_instrument_id(ex['id'])
  ex['audio'] = audio_io.wav_data_to_samples_librosa(
      ex['audio'], sample_rate=audio_sample_rate)
  expected_seconds = ex['audio'].shape[0] / audio_sample_rate
  ex.update(extract_notes(ex['sequence'], expected_seconds))
  beam.metrics.Metrics.distribution('prepare-urmp',
                                    'orig-audio-len').update(len(ex['audio']))
  return ex


def normalize_audio(ex, max_audio):
  ex['audio'] /= max_audio
  return ex


def resample(ex, ddsp_sample_rate, audio_sample_rate):
  """Resample features to standard DDSP sample rate."""
  f0_times = ex['f0_time']
  f0_orig = ex['f0_hz']
  max_time = np.max(f0_times)
  new_times = np.linspace(0, max_time, int(ddsp_sample_rate * max_time))
  if f0_times[0] > 0:
    f0_orig = np.insert(f0_orig, 0, f0_orig[0])
    f0_times = np.insert(f0_times, 0, 0)
  f0_interpolated, _ = melody.resample_melody_series(
      f0_times, f0_orig,
      melody.freq_to_voicing(f0_orig)[1], new_times)
  ex['f0_hz'] = f0_interpolated
  ex['f0_time'] = new_times
  ex['orig_f0_hz'] = f0_orig
  ex['orig_f0_time'] = f0_times

  # Truncate audio to an integer multiple of f0_hz vector.
  num_audio_samples = round(
      len(ex['f0_hz']) * (audio_sample_rate / ddsp_sample_rate))
  beam.metrics.Metrics.distribution(
      'prepare-urmp',
      'resampled-audio-diff').update(num_audio_samples - len(ex['audio']))

  ex['audio'] = ex['audio'][:num_audio_samples]

  # Truncate pianoroll features to length of f0_hz vector.
  for key in [
      'note_active_frame_indices', 'note_active_velocities', 'note_onsets',
      'note_offsets'
  ]:
    ex[key] = ex[key][:len(ex['f0_hz']), :]

  return ex


def batch_dataset(ex, audio_sample_rate, ddsp_sample_rate):
  """Split features and audio into 4 second sliding windows."""
  batched = []
  for key, vec in ex.items():
    if isinstance(vec, np.ndarray):
      if key == 'audio':
        sampling_rate = audio_sample_rate
      else:
        sampling_rate = ddsp_sample_rate

      frames = heuristics.window_array(vec, sampling_rate, 4.0, 0.25)
      if not batched:
        batched = [{} for _ in range(len(frames))]
      for i, frame in enumerate(frames):
        batched[i][key] = frame

  # once batches are created, replicate ids and metadata over all elements.
  for key, val in ex.items():
    if not isinstance(val, np.ndarray):
      for batch in batched:
        batch[key] = val

  beam.metrics.Metrics.counter('prepare-urmp',
                               'batches-created').inc(len(batched))
  return batched


def attach_ddsp_features(ex):
  ex['loudness_db'] = ddsp.spectral_ops.compute_loudness(ex['audio'])
  ex['power_db'] = ddsp.spectral_ops.compute_power(ex['audio'], frame_size=256)
  # ground truth annotations are set with confidence 1.0
  ex['f0_confidence'] = np.ones_like(ex['f0_hz'])
  beam.metrics.Metrics.counter('prepare-urmp', 'ddsp-features-attached').inc()
  return ex


def serialize_tfexample(ex):
  """Creates a tf.Example message ready to be written to a file."""

  def _feature(arr):
    """Returns a feature from a numpy array or string."""
    if isinstance(arr, (bytes, str)):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[arr]))
    else:
      arr = np.asarray(arr).reshape(-1)
      return tf.train.Feature(float_list=tf.train.FloatList(value=arr))

  # Create a dictionary mapping the feature name to the tf.Example-compatible
  # data type.
  feature = {k: _feature(v) for k, v in ex.items()}

  # Create a Features message using tf.train.Example.
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto


def prepare_urmp(input_filepath,
                 output_filepath,
                 instrument_keys,
                 num_shards,
                 batch,
                 force_monophonic,
                 pipeline_options,
                 ddsp_sample_rate=DDSP_SAMPLE_RATE,
                 audio_sample_rate=AUDIO_SAMPLE_RATE):
  """Pipeline for parsing URMP dataset to a usable format for DDSP."""
  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      pipeline_options)
  with beam.Pipeline(options=pipeline_options) as pipeline:
    examples = (
        pipeline
        |
        'read_tfrecords' >> beam.io.tfrecordio.ReadFromTFRecord(input_filepath)
        | 'parse_example' >> beam.Map(parse_example)
        | 'attach_metadata' >> beam.Map(
            attach_metadata,
            ddsp_sample_rate=ddsp_sample_rate,
            audio_sample_rate=audio_sample_rate,
            force_monophonic=force_monophonic))

    if instrument_keys:
      examples |= 'filter_instruments' >> beam.Filter(
          lambda ex: ex['instrument_id'].decode() in instrument_keys)

    examples |= 'resample' >> beam.Map(
        resample,
        ddsp_sample_rate=ddsp_sample_rate,
        audio_sample_rate=audio_sample_rate)
    if batch:
      examples |= 'batch' >> beam.FlatMap(
          batch_dataset,
          audio_sample_rate=audio_sample_rate,
          ddsp_sample_rate=ddsp_sample_rate)
    _ = (
        examples
        | 'attach_ddsp_features' >> beam.Map(attach_ddsp_features)
        | 'filter_silence' >>
        beam.Filter(lambda ex: np.any(ex['loudness_db'] > -70))
        | 'serialize_tfexamples' >> beam.Map(serialize_tfexample)
        | 'shuffle' >> beam.Reshuffle()
        | beam.io.tfrecordio.WriteToTFRecord(
            output_filepath,
            num_shards=num_shards,
            coder=beam.coders.ProtoCoder(tf.train.Example)))
