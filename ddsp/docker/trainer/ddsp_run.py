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
r"""Train, evaluate, or sample (from) a ddsp model.
Usage:
================================================================================
For training, you need to specify --gin_file for both the model and the dataset.
You can optionally specify additional params with --gin_param.
The pip install installs a `ddsp_run` script that can be called directly.
================================================================================
ddsp_run \
--mode=train \
--alsologtostderr \
--save_dir=~/tmp/$USER-ddsp-0 \
--gin_file=models/ae.gin \
--gin_file=datasets/nsynth.gin \
--gin_param=batch_size=16
================================================================================
For evaluation and sampling, only the dataset file is required.
================================================================================
ddsp_run \
--mode=eval \
--alsologtostderr \
--save_dir=~/tmp/$USER-ddsp-0 \
--gin_file=datasets/nsynth.gin
ddsp_run \
--mode=sample \
--alsologtostderr \
--save_dir=~/tmp/$USER-ddsp-0 \
--gin_file=datasets/nsynth.gin
================================================================================
The directory `gin/papers/` stores configs that give the specific models and
datasets used for a paper's experiments, so only require one gin file to train.
================================================================================
ddsp_run \
--mode=train \
--alsologtostderr \
--save_dir=~/tmp/$USER-ddsp-0 \
--gin_file=papers/iclr2020/nsynth_ae.gin
"""

import os
import time

from absl import app
from absl import flags
from absl import logging
from ddsp.training import eval_util
from ddsp.training import models
from ddsp.training import trainers
import gin
import pkg_resources
import tensorflow.compat.v2 as tf

import helper_functions
import magenta_ddsp_internals.train_util as train_util

FLAGS = flags.FLAGS

# Program flags.
flags.DEFINE_enum('mode', 'train', ['train', 'eval', 'sample'],
                  'Whether to train, evaluate, or sample from the model.')
flags.DEFINE_string('save_dir', '~/tmp/ddsp',
                    'Path where checkpoints and summary events will be saved '
                    'during training and evaluation.')
flags.DEFINE_string('restore_dir', '',
                    'Path from which checkpoints will be restored before '
                    'training. Can be different than the save_dir.')
flags.DEFINE_string('tpu', '', 'Address of the TPU. No TPU if left blank.')
flags.DEFINE_multi_string('gpu', [],
                          'Addresses of GPUs for sync data-parallel training.'
                          'Only needs to be specified for using multiple GPUs.')
flags.DEFINE_boolean('allow_memory_growth', False,
                     'Whether to grow the GPU memory usage as is needed by the '
                     'process. Prevents crashes on GPUs with smaller memory.')

flags.DEFINE_enum('strategy', 'one_worker', ['one_worker', 'multiple_workers'],
                  'Wheter to train using one worker or multiple workers')

# Gin config flags.
flags.DEFINE_multi_string('gin_search_path', [],
                          'Additional gin file search paths.')
flags.DEFINE_multi_string('gin_file', [], 'List of paths to the config files.')
flags.DEFINE_multi_string('gin_param', [],
                          'Newline separated list of Gin parameter bindings.')

# Evaluation/sampling specific flags.
flags.DEFINE_boolean('run_once', False, 'Whether evaluation will run once.')
flags.DEFINE_integer('initial_delay_secs', None,
                     'Time to wait before evaluation starts')

GIN_PATH = pkg_resources.resource_filename(__name__, 'gin')

LAST_OPERATIVE_CONFIG_PATH = '/root/trainer/gin/last_config.gin'


def delay_start():
  """Optionally delay the start of the run."""
  delay_time = FLAGS.initial_delay_secs
  if delay_time:
    logging.info('Waiting for %i second(s)', delay_time)
    time.sleep(delay_time)


def parse_gin(restore_dir):
  """Parse gin config from --gin_file, --gin_param, and the model directory."""
  # Add user folders to the gin search path.
  for gin_search_path in [GIN_PATH] + FLAGS.gin_search_path:
    gin.add_config_file_search_path(gin_search_path)

  # Parse gin configs, later calls override earlier ones.
  with gin.unlock_config():
    # Optimization defaults.
    use_tpu = bool(FLAGS.tpu)
    opt_default = 'base.gin' if not use_tpu else 'base_tpu.gin'
    gin.parse_config_file(os.path.join('optimization', opt_default))

    # Load operative_config if it exists (model has already trained).
    operative_config = train_util.get_latest_operative_config(restore_dir)
    if tf.io.gfile.exists(operative_config):
      # Copy the config file from gstorage
      helper_functions.copy_config_file_from_gstorage(operative_config, LAST_OPERATIVE_CONFIG_PATH)
      logging.info('Using operative config: %s', operative_config)
      gin.parse_config_file(operative_config, skip_unknown=True)

    # User gin config and user hyperparameters from flags.
    gin.parse_config_files_and_bindings(
        FLAGS.gin_file, FLAGS.gin_param, skip_unknown=True)


def allow_memory_growth():
  """Sets the GPUs to grow the memory usage as is needed by the process."""
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      # Currently, memory growth needs to be the same across GPUs.
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
      # Memory growth must be set before GPUs have been initialized.
      print(e)


def main(unused_argv):
  """Parse gin config and run ddsp training, evaluation, or sampling."""
  restore_dir = os.path.expanduser(FLAGS.restore_dir)
  save_dir = os.path.expanduser(FLAGS.save_dir)
  # If no separate restore directory is given, use the save directory.
  restore_dir = save_dir if not restore_dir else restore_dir
  logging.info('Restore Dir: %s', restore_dir)
  logging.info('Save Dir: %s', save_dir)

  parse_gin(restore_dir)
  if FLAGS.allow_memory_growth:
    allow_memory_growth()

  # Training.
  if FLAGS.mode == 'train':
    strategy = helper_functions.get_strategy(tpu=FLAGS.tpu, gpus=FLAGS.gpu)

    logging.info('Strategy: %s', restore_dir)
    with strategy.scope():
      model = models.get_model()
      trainer = trainers.Trainer(model, strategy)

    train_util.train(data_provider=gin.REQUIRED,
                     trainer=trainer,
                     save_dir=save_dir,
                     restore_dir=restore_dir)

  # Evaluation.
  elif FLAGS.mode == 'eval':
    model = models.get_model()
    delay_start()
    eval_util.evaluate(data_provider=gin.REQUIRED,
                       model=model,
                       save_dir=save_dir,
                       restore_dir=restore_dir,
                       run_once=FLAGS.run_once)

  # Sampling.
  elif FLAGS.mode == 'sample':
    model = models.get_model()
    delay_start()
    eval_util.sample(data_provider=gin.REQUIRED,
                     model=model,
                     save_dir=save_dir,
                     restore_dir=restore_dir,
                     run_once=FLAGS.run_once)


def console_entry_point():
  """From pip installed script."""
  app.run(main)


if __name__ == '__main__':
  console_entry_point()
