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

r"""Script for gathering input, building the image and submitting job.

Usage:
================================================================================
You can specify all input using flags, then the script triggers building,
pushing the image and submitting the job automatically.

You can specify only a part of input using flags, then the script
asks for missing parameters.

The pip install installs a `ddsp_ai_platform`
script that can be called directly.
================================================================================

ddsp_ai_platform \
--data_path=gs://ddsp_training/data \
--save_dir=gs://ddsp_training/model \
--batch_size=128\
--learning_rate=0.001
"""

import datetime
import os
import subprocess

from absl import app
from absl import flags
from google.cloud import storage

# Program flags.
flags.DEFINE_string('data_path', '',
                    'Path where the dataset for training is saved')
flags.DEFINE_string('save_dir', '',
                    'Path where checkpoints and summary events will be saved '
                    'during training and evaluation.')
flags.DEFINE_string('restore_dir', '',
                    'Path from which checkpoints will be restored before '
                    'training. Can be different than the save_dir.')
flags.DEFINE_string('config_path', '',
                    'Configuration file for AI Platform training')
flags.DEFINE_string('project_id', '', 'GCP Project ID')
flags.DEFINE_string('region', '', 'GCP region for running the training')
flags.DEFINE_string('batch_size', '', 'Batch size')
flags.DEFINE_string('learning_rate', '', 'Learning rate')
flags.DEFINE_string('num_steps', '', 'Number of training steps')
flags.DEFINE_string('steps_per_summary', '', 'Steps per summary')
flags.DEFINE_string('steps_per_save', '', 'Steps per save')
flags.DEFINE_string('early_stop_loss_value', '',
                    'Early stopping. When the total_loss reaches below this '
                    'value training stops.')

FLAGS = flags.FLAGS


def prompt_gs_path(message, required=False):
  required_msg = ' (Required): ' if required else ': '
  message = f'{message}{required_msg}'
  path = input(message)
  if required or path:
    path = check_bucket(path, message)
  return path


def prompt_local_path(message):
  path = input(message)
  if path:
    path = check_local_file(path, message)
  else:
    path = './config_single_vm.yaml'
  return path


def prompt_value(message, default_value):
  message = f'{message} (Default: {default_value}): '
  value = input(message)
  if not value:
    value = default_value
  return value


def check_local_file(file_path, message):
  while not os.path.isfile(file_path):
    print(f'File {file_path} doesn\'t exist.')
    file_path = input(message)
  return file_path


def check_bucket(bucket_name, message):
  """Check if GCS bucket path is valid."""
  while not bucket_name.startswith('gs://'):
    print(f'Path {bucket_name} is not a valid GCS bucket. '
          'Bucket paths must start with gs://.')
    bucket_name = input(message)

  while True:
    try:
      name = bucket_name.split('gs://')[1]
      if '/' in name:
        name = name.split('/')[0]
      storage.Client().get_bucket(name)
      break

    except:  # pylint: disable=bare-except
      print(f'Bucket {bucket_name} doesn\'t exist or can\'t be accessed.')
      bucket_name = input(message)

  return bucket_name


def check_project_id(project_id):
  check_command = 'gcloud projects describe ' + project_id
  while not subprocess.getoutput(check_command).startswith('createTime'):
    print(f'Project {project_id} doesn\'t exist or can\'t be accessed.')
    project_id = input('Project ID: ')
    check_command = 'gcloud projects describe ' + project_id
  return project_id


def get_project_id():
  project_id = subprocess.getoutput('gcloud config get-value project')
  if project_id == '(unset)':
    project_id = input('Project ID: ')
    project_id = check_project_id(project_id)
  return project_id


def get_region():
  region = subprocess.getoutput('gcloud config get-value compute/region')
  if region == '(unset)':
    region = prompt_value('Region for training the model', 'europe-west4')
  return region


def get_input():
  """Gathers input from user."""
  msg = 'Path to training dataset directory'
  if FLAGS.data_path:
    data_path = check_bucket(FLAGS.data_path, msg)
  else:
    data_path = prompt_gs_path(msg, required=True)

  msg = 'Path for saving model, snapshots and summaries'
  if FLAGS.save_dir:
    save_dir = check_bucket(FLAGS.save_dir, msg)
  else:
    save_dir = prompt_gs_path(msg, required=True)

  msg = 'Path for restoring checkpoints before training'
  if not FLAGS.restore_dir:
    restore_dir = prompt_gs_path(msg)
    if not restore_dir:
      restore_dir = save_dir

  msg = 'Path to configuration file: '
  if FLAGS.config_path:
    config_path = check_local_file(FLAGS.config_path, msg)
  else:
    config_path = prompt_local_path(msg)

  if FLAGS.project_id:
    project_id = check_project_id(FLAGS.project_id)
  else:
    project_id = get_project_id()

  image_uri = 'gcr.io/' + project_id + '/ddsp_training:train_job'

  time_diff = datetime.datetime.now() - datetime.datetime(1970, 1, 1)
  uid = int((time_diff).total_seconds())
  job_name = f'training_job_{uid}'

  if not FLAGS.region:
    region = get_region()

  msg = 'Batch size'
  if FLAGS.batch_size:
    batch_size = FLAGS.batch_size
  else:
    batch_size = prompt_value(msg, '16')

  msg = 'Learning rate'
  if FLAGS.learning_rate:
    learning_rate = FLAGS.learning_rate
  else:
    learning_rate = prompt_value(msg, '0.0001')

  msg = 'Number of steps'
  if FLAGS.num_steps:
    num_steps = FLAGS.num_steps
  else:
    num_steps = prompt_value(msg, '40000')

  msg = 'Steps per save'
  if FLAGS.steps_per_save:
    steps_per_save = FLAGS.steps_per_save
  else:
    steps_per_save = prompt_value(msg, '300')

  msg = 'Steps per summary'
  if FLAGS.steps_per_summary:
    steps_per_summary = FLAGS.steps_per_summary
  else:
    steps_per_summary = prompt_value(msg, '300')

  msg = 'Early stop loss value'
  if FLAGS.early_stop_loss_value:
    early_stop_loss_value = FLAGS.early_stop_loss_value
  else:
    early_stop_loss_value = prompt_value(msg, '5')

  args = {'data_path': data_path,
          'save_dir': save_dir,
          'restore_dir': restore_dir,
          'config_path': config_path,
          'image_uri': image_uri,
          'job_name': job_name,
          'region': region,
          'batch_size': batch_size,
          'learning_rate': learning_rate,
          'num_steps': num_steps,
          'steps_per_save': steps_per_save,
          'steps_per_summary': steps_per_summary,
          'early_stop_loss_value': early_stop_loss_value}
  return args


def build_image(args):
  """Builds the docker image."""
  build_command = f'docker build -f Dockerfile -t {args["image_uri"]} ./'
  os.system(build_command)


def push_image(args):
  """Pushes the docker image on Google Cloud Registry."""
  pushing_image = f'docker push {args["image_uri"]}'
  os.system(pushing_image)


def submit_job(args):
  """Submits the job on AI Platform."""

  submitting_job = (
      'gcloud ai-platform jobs submit training'
      f' {args["job_name"]}'
      f' --region={args["region"]}'
      f' --master-image-uri={args["image_uri"]}'
      f' --config={args["config_path"]}'
      f' -- --save_dir={args["save_dir"]}'
      f' --restore_dir={args["restore_dir"]}'
      f' --file_pattern={args["data_path"]}/train.tfrecord*'
      f' --batch_size={args["batch_size"]}'
      f' --learning_rate={args["learning_rate"]}'
      f' --num_steps={args["num_steps"]}'
      f' --steps_per_summary={args["steps_per_summary"]}'
      f' --steps_per_save={args["steps_per_save"]}'
      f' --early_stop_loss_value={args["early_stop_loss_value"]}')

  os.system(submitting_job)


def enable_tensorboard(args):
  """Enables Tensorboard."""
  os.system('gcloud auth login')
  tensorboard_command = f'tensorboard --logdir={args["save_dir"]} --port=6006 &'
  os.system(tensorboard_command)


def upload_logs(args):
  """Uploads logs to TensorBoard.dev."""
  tensorboard_dev_command = ('tensorboard dev upload ' +
                             f'--logdir={args["save_dir"]}' +
                             f' --name \"{args["job_name"]}\"')
  os.system(tensorboard_dev_command)


def main(unused_argv):
  """Gathers input, submits job and enables TensorBoard."""
  args = get_input()

  build_image(args)
  print('Docker image built')

  push_image(args)
  print('Image pushed to Google Cloud Registry')

  submit_job(args)
  print('Job submitted to AI Platform')

  enable_tensorboard(args)
  print('Tensorboard enabled')

  upload_logs(args)
  print('Logs uploaded to TensorBoard.dev')


def console_entry_point():
  """From pip installed script."""
  app.run(main)


if __name__ == '__main__':
  console_entry_point()
