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
"""Script for gathering input, building the image and submitting job.

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

import os
import datetime
import subprocess

from absl import app
from absl import flags
from google.cloud import storage

FLAGS = flags.FLAGS

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

def prompt(default, message, skip=False, arg='', local=False):
  input_prompt = 'Insert '
  if default == 'path':
    input_prompt += 'the path to where the '
    input_prompt += message
    if not local:
      input_prompt += '. This must be a GCS Bucket'
    if skip:
      input_prompt += '. Skip for the default one'
    input_prompt += ':\n'
  if default == 'value':
    input_prompt += message
    if skip:
      input_prompt += ' or skip for the default value (' + arg + ')'
    input_prompt += ':'
  return input_prompt

def check_bucket(bucket_name, message):
  while not bucket_name.startswith('gs://'):
    print('Path not a bucket.')
    bucket_name = input(message)
  while True:
    try:
      name = bucket_name[5:]
      if '/' in name:
        name = name.split('/')[0]
      storage.Client().get_bucket(name)
      break
    except:
      print('Not a valid bucket.')
      bucket_name = input(message)
  return bucket_name

def get_input():
  """Gathers input from user"""
  # Determining path for data retrieval
  input_dict = dict()
  if FLAGS.data_path:
    data_path = FLAGS.data_path
  else:
    data_path = input(prompt('path', 'dataset for training is stored'))
  data_path = check_bucket(data_path,
                           prompt('path', 'dataset for training is stored'))
  input_dict['data_path'] = data_path

  # Determining path for storing model, snapshots and summaries
  if FLAGS.save_dir:
    save_dir = FLAGS.save_dir
  else:
    save_dir = input(prompt('path', 'model, '
                            'snapshots and summaries will be saved'))
  save_dir = check_bucket(save_dir, prompt('path', 'model, snapshots '
                                           'and summaries will be saved'))
  input_dict['save_dir'] = save_dir

  # Determining path for restoring the model
  if FLAGS.restore_dir:
    restore_dir = FLAGS.restore_dir
  else:
    if FLAGS.save_dir:
      restore_dir = FLAGS.save_dir
    else:
      restore_dir = input(prompt('path', 'checkpoints will be restored '
                                 'from before training', skip=True))
      if restore_dir == '':
        restore_dir = save_dir
  restore_dir = check_bucket(restore_dir, prompt('path', 'checkpoints will be '
                                                 'restored from before '
                                                 'training', skip=True))
  input_dict['restore_dir'] = restore_dir

  if FLAGS.config_path:
    config_path = FLAGS.config_path
  else:
    config_path = input(prompt('path', 'configuration file is stored',\
                               skip=True, local=True))
    if config_path == '':
      config_path = './config_multiple_vms.yaml'
  input_dict['config_path'] = config_path

  if FLAGS.project_id:
    project_id = FLAGS.project_id
  else:
    project_id = subprocess.getoutput('gcloud config get-value project')
    if project_id == '(unset)':
      project_id = input(prompt('value', 'your project ID'))
    check_project_id = 'gcloud projects describe ' + project_id
    while not subprocess.getoutput(check_project_id).startswith('createTime'):
      print('Project ID not valid.')
      project_id = input(prompt('value', 'your project ID'))
      check_project_id = 'gcloud projects describe ' + project_id

  image_uri = 'gcr.io/' + project_id + '/ddsp_training:train_job'
  input_dict['image_uri'] = image_uri

  job_name = 'training_job_' + str(int((datetime.datetime.now()\
             - datetime.datetime(1970,1,1)).total_seconds()))
  input_dict['job_name'] = job_name

  if FLAGS.region:
    region = FLAGS.region
  else:
    region = subprocess.getoutput('gcloud config get-value compute/region')
    if region == '(unset)':
      region = input('\nInsert the region you want to train '
                      'your model in or skip for the default '
                      'value (europe-west4): ')
      if region == '':
        region = 'europe-west4'
  input_dict['region'] = region

  if FLAGS.batch_size:
    batch_size = FLAGS.batch_size
  else:
    batch_size = input(prompt('value', 'batch size', skip=True, arg='128'))
    if batch_size == '':
      batch_size = '128'
  input_dict['batch_size'] = batch_size

  if FLAGS.learning_rate:
    learning_rate = FLAGS.learning_rate
  else:
    learning_rate = input(prompt('value', 'the learning rate',\
                                 skip=True, arg='0.001'))
    if learning_rate == '':
      learning_rate = '0.001'
  input_dict['learning_rate'] = learning_rate

  if FLAGS.num_steps:
    num_steps = FLAGS.num_steps
  else:
    num_steps = input(prompt('value', 'the number of steps for training',\
                             skip=True, arg='15000'))
    if num_steps == '':
      num_steps = '15000'
  input_dict['num_steps'] = num_steps

  if FLAGS.steps_per_save:
    steps_per_save = FLAGS.steps_per_save
  else:
    steps_per_save = input(prompt('value', 'the number of steps per save',\
                                  skip=True, arg='300'))
    if steps_per_save == '':
      steps_per_save = '300'
  input_dict['steps_per_save'] = steps_per_save

  if FLAGS.steps_per_summary:
    steps_per_summary = FLAGS.steps_per_summary
  else:
    steps_per_summary = input(prompt('value', 'the number of steps per'
                                     'summary', skip=True, arg='300'))
    if steps_per_summary == '':
      steps_per_summary = '300'
  input_dict['steps_per_summary'] = steps_per_summary

  if FLAGS.early_stop_loss_value:
    early_stop_loss_value = FLAGS.early_stop_loss_value
  else:
    early_stop_loss_value = input(prompt('value', 'the early stop loss value',\
                                         skip=True, arg='5'))
    if early_stop_loss_value == '':
      early_stop_loss_value = '5'
  input_dict['early_stop_loss_value'] = early_stop_loss_value

  return input_dict

def build_image(input_dict):
  """Builds the docker image"""
  build_command = 'docker build -f Dockerfile -t '\
                  + input_dict['image_uri'] + ' ./'
  os.system(build_command)

def push_image(input_dict):
  """Pushes the docker image on Google Cloud Registry"""
  pushing_image = 'docker push ' + input_dict['image_uri']
  os.system(pushing_image)

def submit_job(input_dict):
  """Submits the job on AI Platform"""
  os.system('export PATH=/usr/local/google/home/$USER/.local/bin:$PATH')

  submitting_job = 'gcloud beta ai-platform jobs submit training '\
  + input_dict['job_name'] + ' --region ' + input_dict['region']\
  + ' --master-image-uri ' + input_dict['image_uri']\
  + ' --config ' + input_dict['config_path']\
  + ' -- --save_dir=' + input_dict['save_dir']\
  + ' --restore_dir=' + input_dict['restore_dir']\
  + ' --file_pattern=' + input_dict['data_path'] + '/train.tfrecord*'\
  + ' --batch_size=' + input_dict['batch_size']\
  + ' --learning_rate=' + input_dict['learning_rate']\
  + ' --num_steps=' + input_dict['num_steps']\
  + ' --steps_per_summary=' + input_dict['steps_per_summary']\
  + ' --steps_per_save=' + input_dict['steps_per_save']\
  + ' --early_stop_loss_value=' + input_dict['early_stop_loss_value']
  os.system(submitting_job)

def enable_tensorboard(input_dict):
  """Enables Tensorboard"""
  os.system('gcloud auth login')
  tensorboard_command = 'tensorboard --logdir='\
                        + input_dict['save_dir'] + ' --port=8082 &'
  os.system(tensorboard_command)

def upload_logs(input_dict):
  """Uploads logs to TensorBoard.dev"""
  tensorboard_dev_command = 'tensorboard dev upload --logdir '\
                            + input_dict['save_dir']\
  + ' --name \"' + input_dict['job_name'] + '\"'
  os.system(tensorboard_dev_command)

def main(unused_argv):
  """Gathers input, submits job and enables TensorBoard"""
  input_dict = get_input()

  build_image(input_dict)
  print('Docker image built')

  push_image(input_dict)
  print('Image pushed to Google Cloud Registry')

  submit_job(input_dict)
  print('Job submitted to AI Platform')

  enable_tensorboard(input_dict)
  print('Tensorboard enabled')

  upload_logs(input_dict)
  print('Logs uploaded to TensorBoard.dev')

def console_entry_point():
  """From pip installed script."""
  app.run(main)

if __name__ == '__main__':
  console_entry_point()
