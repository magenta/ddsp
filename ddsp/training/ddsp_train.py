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
ddsp_train \
--use_tpu=false \
--alsologtostderr \
--model_dir=~/tmp/$USER-ddsp-0 \
--gin_file=iclr2020/nsynth_ae.gin

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from ddsp.training import train_util
import gin
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

FLAGS = flags.FLAGS
GIN_PATH = os.path.join(os.path.dirname(__file__), 'gin')

flags.DEFINE_string('model_dir', '~/tmp/ddsp',
                    'Path where checkpoints and summary events will be located '
                    'during training and evaluation.')
flags.DEFINE_integer('num_steps', 1000000,
                     'Number of training steps or `None` for infinite.')
flags.DEFINE_string('master', '', 'Name of the TensorFlow runtime to use.')
flags.DEFINE_boolean('use_tpu', True, 'Whether training will happen on a TPU.')
flags.DEFINE_string('file_pattern', None,
                    'Regex of paths for dataset files to train with.')
flags.DEFINE_multi_string('gin_file', [], 'List of paths to the config files.')
flags.DEFINE_multi_string('gin_param', [],
                          'Newline separated list of Gin parameter bindings.')


def run():
  """Parse gin config and run training."""
  # Hyperparameters: parse gin_files and manual parameters.
  if FLAGS.gin_file is not None:
    gin_file = [os.path.join(GIN_PATH, 'train', f) for f in FLAGS.gin_file]
  with gin.unlock_config():
    fname = 'default_tpu.gin' if FLAGS.use_tpu else 'default_gpu.gin'
    gin.parse_config_file(os.path.join(GIN_PATH, 'optimization', fname))
    gin.parse_config_files_and_bindings(gin_file,
                                        FLAGS.gin_param,
                                        skip_unknown=True)
  # Run training after parsing gin config.
  train_util.train(model_dir=FLAGS.model_dir,
                   num_steps=FLAGS.num_steps,
                   master=FLAGS.master,
                   use_tpu=FLAGS.use_tpu,
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
