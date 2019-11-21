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
r"""Evaluate ddsp model.

Example Local Usage:
====================
ddsp_eval \
--gin_file=nsynth/default_eval.gin \
--alsologtostderr \
--model_dir=~/tmp/ddsp/$USER-ddsp-0 \
--run_once=True \
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
import gin
import tensorflow.compat.v1 as tf


FLAGS = flags.FLAGS
GIN_PATH = os.path.join(os.path.dirname(__file__), 'gin')

flags.DEFINE_string(
    'model_dir', '~/tmp/ddsp/training',
    'Path where checkpoints and summary events will be located during '
    'training and evaluation.')
flags.DEFINE_string('master', '', 'Name of the TensorFlow runtime to use.')
flags.DEFINE_boolean('run_once', False, 'Whether evaluation will run once.')
flags.DEFINE_string('file_pattern', None, 'Regex of dataset files to use.')
flags.DEFINE_integer('initial_delay_secs', None,
                     'Time to wait before evaluation starts')
flags.DEFINE_multi_string('gin_file', [], 'List of paths to the config files.')
flags.DEFINE_multi_string('gin_param', [],
                          'Newline separated list of Gin parameter bindings.')


def run():
  """Run evaluation."""
  if FLAGS.initial_delay_secs:
    logging.info('Waiting for %i second(s)', FLAGS.initial_delay_secs)
    time.sleep(FLAGS.initial_delay_secs)

  tf.reset_default_graph()

  # Parse FLAGS
  model_dir = os.path.expanduser(FLAGS.model_dir)

  # Hyperparameters: parse gin_files and manual parameters.
  train_gin_file = [os.path.join(model_dir, 'operative_config-0.gin')]
  with gin.unlock_config():
    gin.parse_config_files_and_bindings(train_gin_file, None, skip_unknown=True)
  if FLAGS.gin_file is not None:
    eval_gin_file = [os.path.join(GIN_PATH, 'eval', f) for f in FLAGS.gin_file]
  with gin.unlock_config():
    gin.parse_config_files_and_bindings(eval_gin_file,
                                        FLAGS.gin_param,
                                        skip_unknown=True)
  # Run evaluation loop.
  eval_util.evaluate_or_sample(model_dir=model_dir,
                               master=FLAGS.master,
                               run_once=FLAGS.run_once,
                               file_pattern=FLAGS.file_pattern)


def main(unused_argv):
  """From command line."""
  run()


def console_entry_point():
  """From pip installed script."""
  tf.disable_v2_behavior()
  app.run(main)


if __name__ == '__main__':
  console_entry_point()
