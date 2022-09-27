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

r"""Prepare URMP dataset DDSP and NoteSequence features.

Usage:
====================
ddsp_prepare_urmp_dataset \
  --input_filepath='/path/to/input.tfrecord-*' \
  --output_filepath='/path/to/output.tfrecord' \
  --instrument_key=vn \
  --num_shards=10 \
  --alsologtostderr

"""

from absl import app
from absl import flags

from ddsp.training.data_preparation.prepare_urmp_dataset_lib import prepare_urmp
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('input_filepath', '', 'Input filepath for dataset.')
flags.DEFINE_string('output_filepath', '', 'Output filepath for dataset.')
flags.DEFINE_multi_string(
    'instrument_key', [], 'Instrument keys to extract. '
    'If not set, extract all instruments. Possible keys '
    'are vn, va, vc, db, fl, ob, cl, sax, bn, tpt, hn, '
    'tbn, tba.')
flags.DEFINE_integer(
    'num_shards', None, 'Num shards for output dataset. If '
    'None, this number will be determined automatically.')
flags.DEFINE_bool('batch', True, 'Whether or not to batch the dataset.')
flags.DEFINE_bool('force_monophonic', True, 'Fix URMP note labels such that '
                  'note onsets and offsets do not overlap.')
flags.DEFINE_list(
    'pipeline_options', '--runner=DirectRunner',
    'A comma-separated list of command line arguments to be used as options '
    'for the Beam Pipeline.')
flags.DEFINE_integer('ddsp_sample_rate', 250, 'Sample rate for dataset output.')
flags.DEFINE_integer('audio_sample_rate', 16000, 'Sample rate for URMP audio.')


def run():
  prepare_urmp(
      input_filepath=FLAGS.input_filepath,
      output_filepath=FLAGS.output_filepath,
      instrument_keys=FLAGS.instrument_key,
      num_shards=FLAGS.num_shards,
      batch=FLAGS.batch,
      force_monophonic=FLAGS.force_monophonic,
      pipeline_options=FLAGS.pipeline_options,
      ddsp_sample_rate=FLAGS.ddsp_sample_rate,
      audio_sample_rate=FLAGS.audio_sample_rate)


def main(unused_argv):
  """From command line."""
  run()


def console_entry_point():
  """From pip installed script."""
  app.run(main)


if __name__ == '__main__':
  console_entry_point()
