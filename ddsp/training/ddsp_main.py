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
r"""Train ddsp model.

Usage:
====================
dds_main \
--mode=train \
--use_tpu=false \
--alsologtostderr \
--model_dir=~/tmp/$USER-ddsp-0 \
--gin_file=train/iclr2020/nsynth_ae.gin

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import app
from absl import flags
from absl import logging
from ddsp.training import eval_util
from ddsp.training import train_util
import gin
import pkg_resources
import tensorflow.compat.v1 as tf

# Imports for gin.
# pylint:disable=unused-import,g-bad-import-order
import ddsp.training.data
import ddsp.training.model
# pylint:enable=unused-import,g-bad-import-order

tf.disable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_enum('mode', 'train', ['train', 'eval'],
                  'Whether to train or evaluate the model.')
flags.DEFINE_string('model_dir', '~/tmp/ddsp',
                    'Path where checkpoints and summary events will be located '
                    'during training and evaluation.')
flags.DEFINE_string('master', '', 'Name of the TensorFlow runtime to use.')
flags.DEFINE_boolean('use_tpu', True, 'Whether training will happen on a TPU.')

# Dataset flags.
flags.DEFINE_string(
    'file_pattern', None,
    'Regex of dataset files to use with a TFRecordProvider.')
flags.DEFINE_string(
    'tfds_name', None,
    'TFDS name to use with a TfdsProvider.')
flags.DEFINE_string(
    'tfds_split', None,
    'Split to use with a TfdsProvider.')
flags.DEFINE_string(
    'tfds_data_dir', None,
    'TFDS data directory to read from with a TfdsProvider.')

# Gin config flags.
flags.DEFINE_multi_string(
    'gin_search_path', [], 'Additional gin file search paths.')
flags.DEFINE_multi_string('gin_file', [], 'List of paths to the config files.')
flags.DEFINE_multi_string('gin_param', [],
                          'Newline separated list of Gin parameter bindings.')

flags.DEFINE_integer('num_train_steps', 1000000,
                     'Number of training steps or `None` for infinite.')
flags.DEFINE_boolean('eval_once', False, 'Whether evaluation will run once.')
flags.DEFINE_integer('initial_delay_secs', None,
                     'Time to wait before evaluation starts')

GIN_PATH = pkg_resources.resource_filename(__name__, 'gin')


def run():
  """Parse gin config and run training."""
  # Hyperparameters: parse gin_files and manual parameters.
  for gin_search_path in [GIN_PATH] + FLAGS.gin_search_path:
    gin.add_config_file_search_path(gin_search_path)
  with gin.unlock_config():
    fname = 'default_tpu.gin' if FLAGS.use_tpu else 'default_gpu.gin'
    gin.parse_config_file(os.path.join('optimization', fname))
    gin.parse_config_files_and_bindings(FLAGS.gin_file,
                                        FLAGS.gin_param,
                                        skip_unknown=True)

  model_dir = os.path.expanduser(FLAGS.model_dir)

  dataset_kwargs = {}
  for dataset_flag in (
      'file_pattern', 'tfds_name', 'tfds_split', 'tfds_data_dir'):
    if FLAGS[dataset_flag].value:
      dataset_kwargs[dataset_flag] = FLAGS[dataset_flag].value

  if FLAGS.mode == 'train':
    # Run training after parsing gin config.
    train_util.train(model_dir=model_dir,
                     num_steps=FLAGS.num_train_steps,
                     master=FLAGS.master,
                     use_tpu=FLAGS.use_tpu,
                     dataset_kwargs=dataset_kwargs)
  elif FLAGS.mode == 'eval':
    if FLAGS.initial_delay_secs:
      logging.info('Waiting for %i second(s)', FLAGS.initial_delay_secs)
      time.sleep(FLAGS.initial_delay_secs)
    eval_util.evaluate_or_sample(model_dir=model_dir,
                                 master=FLAGS.master,
                                 run_once=FLAGS.eval_once,
                                 dataset_kwargs=dataset_kwargs)


def main(unused_argv):
  """From command line."""
  run()


def console_entry_point():
  """From pip installed script."""
  tf.disable_v2_behavior()
  app.run(main)


if __name__ == '__main__':
  console_entry_point()
