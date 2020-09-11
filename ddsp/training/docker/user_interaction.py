#!/usr/bin/env python3
"""Script for gathering input, building the image and submitting job"""

import os

def get_input():
  """Gathers input from user"""
  # Determining path for data retrieval
  input_dict = dict()
  data_path = input('Insert a path for data retrieving. '
                      'This must be a GCS Bucket:\n')
  input_dict['data_path'] = data_path

  # Determining path for storing model, snapshots and summaries
  save_dir = input('\nInsert a path for saving the model, '
                    'snapshots and summaries. This must be a GCS Bucket:\n')
  input_dict['save_dir'] = save_dir

  # Determining path for restoring the model
  restore_dir = input('\nInsert a path from which checkpoints will '
                      'be restored before training. Skip for using '
                      'the same path as for saving:\n')
  if restore_dir == '':
    restore_dir = save_dir
  input_dict['restore_dir'] = restore_dir

  config_path = input('\nInsert the path to a configuration file '
                      'or skip for the default one:\n')
  if config_path == '':
    config_path = './config_single_vm.yaml'
  input_dict['config_path'] = config_path

  image_uri = input('\nInsert an IMAGE URI. '
                    'The template is: gcr.io/<GCP_PROJECT_ID>'
                    '/<IMAGE_REPO_NAME>:<IMAGE_TAG> :\n')
  input_dict['image_uri'] = image_uri

  job_name = input('\nInsert a job name:\n')
  input_dict['job_name'] = job_name

  region = input('\nInsert the region you want to train '
                  'your model in or skip for the default '
                  'value (europe-west4): ')
  if region == '':
    region = 'europe-west4'
  input_dict['region'] = region

  batch_size = input('\nInsert batch size '
                      'or skip for the default value (128): ')
  if batch_size == '':
    batch_size = '128'
  input_dict['batch_size'] = batch_size

  learning_rate = input('\nInsert the learning rate '
                        'or skip for the default value (0.001): ')
  if learning_rate == '':
    learning_rate = '0.001'
  input_dict['learning_rate'] = learning_rate

  no_of_steps = input('\nInsert the number of steps for training '
                      'or skip for the default value (15000): ')
  if no_of_steps == '':
    no_of_steps = '15000'
  input_dict['no_of_steps'] = no_of_steps

  steps_per_save = input('\nInsert the number of steps per save '
                          'or skip for the default value (300): ')
  if steps_per_save == '':
    steps_per_save = '300'
  input_dict['steps_per_save'] = steps_per_save

  steps_per_summary = input('\nInsert the number of steps per summary '
                            'or skip for the default value (300): ')
  if steps_per_summary == '':
    steps_per_summary = '300'
  input_dict['steps_per_summary'] = steps_per_summary

  early_stop_loss_value = input('\nInsert the early stop loss value '
                                'or skip for the default value (5): ')
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
  + ' --num_steps=' + input_dict['no_of_steps']\
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

def main():
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

if __name__ == '__main__':
  main()
