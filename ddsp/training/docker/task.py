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

"""Script for running a containerized training on Google Cloud AI Platform."""

import json
import os
import subprocess

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('save_dir', None,
                    'Path where checkpoints and summary events will be saved '
                    'during training and evaluation.')
flags.DEFINE_string('restore_dir', '',
                    'Path from which checkpoints will be restored before '
                    'training. Can be different than the save_dir.')
flags.DEFINE_string('file_pattern', None, 'Data file pattern')

flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_float('learning_rate', 0.003, 'Learning rate')
flags.DEFINE_integer('num_steps', 30000, 'Number of training steps')
flags.DEFINE_float('early_stop_loss_value', 0.0,
                   'Early stopping. When the total_loss reaches below this '
                   'value training stops.')

flags.DEFINE_integer('steps_per_summary', 300, 'Steps per summary')
flags.DEFINE_integer('steps_per_save', 300, 'Steps per save')
flags.DEFINE_boolean('hypertune', False,
                     'Enable metric reporting for hyperparameter tuning.')


flags.DEFINE_multi_string('gin_search_path', [],
                          'Additional gin file search paths. '
                          'Must be paths inside Docker container and '
                          'necessary gin configs should be added at the '
                          'Docker image building stage.')
flags.DEFINE_multi_string('gin_file', [],
                          'List of paths to the config files. If file '
                          'in gstorage bucket specify whole gstorage path: '
                          'gs://bucket-name/dir/in/bucket/file.gin. If path '
                          'should be local remember about copying the file '
                          'inside the Docker container at building stage. ')
flags.DEFINE_multi_string('gin_param', [],
                          'Newline separated list of Gin parameter bindings.')


def get_worker_behavior_info(save_dir):
  """Infers worker behavior from the environment.

  Checks if TF_CONFIG environment variable is set
  and inferes cluster configuration and save_dir
  from it.

  Args:
    save_dir: Save directory given by the user.

  Returns:
    cluster_config: Inferred cluster configuration.
    save_dir: Inferred save directory.
  """
  if 'TF_CONFIG' in os.environ:
    cluster_config = os.environ['TF_CONFIG']
    cluster_config_dict = json.loads(cluster_config)
    if ('cluster' not in cluster_config_dict.keys() or
        'task' not in cluster_config_dict or
        len(cluster_config_dict['cluster']) <= 1):
      cluster_config = ''
    elif cluster_config_dict['task']['type'] != 'chief':
      save_dir = ''
  else:
    cluster_config = ''

  return cluster_config, save_dir


def parse_list_params(list_of_params, param_name):
  return [f'--{param_name}={param}' for param in list_of_params]


def main(unused_argv):
  restore_dir = FLAGS.save_dir if not FLAGS.restore_dir else FLAGS.restore_dir

  cluster_config, save_dir = get_worker_behavior_info(FLAGS.save_dir)
  gin_search_path = parse_list_params(FLAGS.gin_search_path, 'gin_search_path')
  gin_file = parse_list_params(FLAGS.gin_file, 'gin_file')
  gin_param = parse_list_params(FLAGS.gin_param, 'gin_param')

  ddsp_run_command = (
      ['ddsp_run',
       '--mode=train',
       '--alsologtostderr',
       '--gin_file=models/solo_instrument.gin',
       '-gin_file=datasets/tfrecord.gin',
       f'--cluster_config={cluster_config}',
       f'--save_dir={save_dir}',
       f'--restore_dir={restore_dir}',
       f'--hypertune={FLAGS.hypertune}',
       f'--early_stop_loss_value={FLAGS.early_stop_loss_value}',
       f'--gin_param=batch_size={FLAGS.batch_size}',
       f'--gin_param=learning_rate={FLAGS.learning_rate}',
       f'--gin_param=TFRecordProvider.file_pattern=\'{FLAGS.file_pattern}\'',
       f'--gin_param=train_util.train.num_steps={FLAGS.num_steps}',
       f'--gin_param=train_util.train.steps_per_save={FLAGS.steps_per_save}',
       ('--gin_param=train_util.train.steps_per_summary='
        f'{FLAGS.steps_per_summary}')]
      + gin_search_path + gin_file + gin_param)

  subprocess.run(args=ddsp_run_command, check=True)

if __name__ == '__main__':
  flags.mark_flag_as_required('file_pattern')
  flags.mark_flag_as_required('save_dir')
  app.run(main)
