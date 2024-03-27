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

r"""Apache Beam pipeline for computing TFRecord dataset of synthetic examples.

Example usage:
==============
ddsp_generate_synthetic_dataset \
--output_tfrecord_path=/tmp/synthetic_data.tfrecord \
--num_shards=1 \
--gin_param="generate_examples.generate_fn = @generate_notes_v2" \
--num_examples=100 \
--alsologtostderr

For the ICML workshop paper, we created 10,000,000 examples in 1000 shards.
"""

from absl import app
from absl import flags
import apache_beam as beam
from ddsp.training.data_preparation import synthetic_data  # pylint:disable=unused-import
import gin
import numpy as np
import pkg_resources
import tensorflow.compat.v2 as tf


GIN_PATH = pkg_resources.resource_filename(__name__, '../gin')

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'output_tfrecord_path', None,
    'The prefix path to the output TFRecord. Shard numbers will be added to '
    'actual path(s).')
flags.DEFINE_integer(
    'num_shards', None,
    'The number of shards to use for the TFRecord. If None, this number will '
    'be determined automatically.')
flags.DEFINE_integer(
    'num_examples', 1000000,
    'The total number of synthetic examples to generate.')
flags.DEFINE_integer(
    'random_seed', 42,
    'Random seed to use for deterministic generation.')
flags.DEFINE_list(
    'pipeline_options', '--runner=DirectRunner',
    'A comma-separated list of command line arguments to be used as options '
    'for the Beam Pipeline.')

# Gin config flags.
flags.DEFINE_multi_string('gin_search_path', [],
                          'Additional gin file search paths.')
flags.DEFINE_multi_string('gin_file', [], 'List of paths to the config files.')
flags.DEFINE_multi_string('gin_param', [],
                          'Newline separated list of Gin parameter bindings.')


class GenerateExampleFn(beam.DoFn):
  """Gin-configurable wrapper to generate synthetic examples."""

  def __init__(self, gin_str, **kwargs):
    super().__init__(**kwargs)
    self._gin_str = gin_str

  def start_bundle(self):
    with gin.unlock_config():
      gin.parse_config(self._gin_str)

  @gin.configurable('generate_examples')
  def process(self, seed, generate_fn=gin.REQUIRED):
    np.random.seed(seed)
    batch = generate_fn(n_batch=1)
    beam.metrics.Metrics.counter('GenerateExampleFn', 'generated').inc()
    yield {k: v[0].numpy() for k, v in batch.items()}


def _float_dict_to_tfexample(float_dict):
  """Convert dictionary of float arrays to tf.train.Example proto."""
  return tf.train.Example(
      features=tf.train.Features(
          feature={
              k: tf.train.Feature(
                  float_list=tf.train.FloatList(value=v.flatten()))
              for k, v in float_dict.items()
          }
      ))


def run():
  """Run the beam pipeline to create synthetic dataset."""
  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      FLAGS.pipeline_options)
  with beam.Pipeline(options=pipeline_options) as pipeline:
    for gin_search_path in [GIN_PATH] + FLAGS.gin_search_path:
      gin.add_config_file_search_path(gin_search_path)
    gin.parse_config_files_and_bindings(
        FLAGS.gin_file, FLAGS.gin_param, skip_unknown=True)

    np.random.seed(FLAGS.random_seed)
    _ = (
        pipeline
        | beam.Create(np.random.randint(2**32, size=FLAGS.num_examples))
        | beam.ParDo(GenerateExampleFn(gin.config_str()))
        | beam.Reshuffle()
        | beam.Map(_float_dict_to_tfexample)
        | beam.io.tfrecordio.WriteToTFRecord(
            FLAGS.output_tfrecord_path,
            num_shards=FLAGS.num_shards,
            coder=beam.coders.ProtoCoder(tf.train.Example))
    )


def main(unused_argv):
  """From command line."""
  run()


def console_entry_point():
  """From pip installed script."""
  app.run(main)


if __name__ == '__main__':
  console_entry_point()
