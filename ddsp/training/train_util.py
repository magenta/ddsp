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

"""Library of training functions."""

import json
import os
import time

from absl import logging
from ddsp.training import cloud
import gin
import tensorflow.compat.v2 as tf



# ---------------------- Helper Functions --------------------------------------
def get_strategy(tpu='', cluster_config=''):
  """Create a distribution strategy for running on accelerators.

  For CPU, single-GPU, or multi-GPU jobs on a single machine, call this function
  without args to return a MirroredStrategy.

  For TPU jobs, specify an address to the `tpu` argument.

  For multi-machine GPU jobs, specify a `cluster_config` argument of the cluster
  configuration.

  Args:
    tpu: Address of the TPU. No TPU if left blank.
    cluster_config: Should be specified only for multi-worker jobs.
      Task specific dictionary for cluster config dict in the TF_CONFIG format.
      https://www.tensorflow.org/guide/distributed_training#setting_up_tf_config_environment_variable
      If passed as a string, will be parsed to a dictionary. Two components
      should be specified: cluster and task. Cluster provides information about
      the training cluster, which is a dict consisting of different types of
      jobs such as chief and worker. Task is information about the current task.
      For example: "{"cluster": {"worker": ["host1:port", "host2:port"]},
                     "task": {"type": "worker", "index": 0}}"

  Returns:
    A distribution strategy. MirroredStrategy by default. TPUStrategy if `tpu`
    arg is specified. MultiWorkerMirroredStrategy if `cluster_config` arg is
    specified.
  """
  if tpu:
    logging.info('Use TPU at %s', tpu)
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
  elif  cluster_config:
    if not isinstance(cluster_config, dict):
      cluster_config = json.loads(cluster_config)
    cluster_spec = tf.train.ClusterSpec(cluster_config['cluster'])
    resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
        cluster_spec=cluster_spec,
        task_type=cluster_config['task']['type'],
        task_id=cluster_config['task']['index'],
        num_accelerators={'GPU': len(tf.config.list_physical_devices('GPU'))},
        rpc_layer='grpc')
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        cluster_resolver=resolver)
  else:
    logging.info('Defaulting to MirroredStrategy')
    strategy = tf.distribute.MirroredStrategy()
  return strategy


def expand_path(file_path):
  return os.path.expanduser(os.path.expandvars(file_path))


def get_latest_file(dir_path, prefix='operative_config-', suffix='.gin'):
  """Returns latest file with pattern '/dir_path/prefix[iteration]suffix'.

  Args:
    dir_path: Path to the directory.
    prefix: Filename prefix, not including directory.
    suffix: Filename suffix, including extension.

  Returns:
    Path to the latest file

  Raises:
    FileNotFoundError: If no files match the pattern
      '/dir_path/prefix[int]suffix'.
  """
  dir_path = expand_path(dir_path)
  dir_prefix = os.path.join(dir_path, prefix)
  search_pattern = dir_prefix + '*' + suffix
  file_paths = tf.io.gfile.glob(search_pattern)
  if not file_paths:
    raise FileNotFoundError(
        f'No files found matching the pattern \'{search_pattern}\'.')
  try:
    # Filter to get highest iteration, no negative iterations.
    get_iter = lambda fp: abs(int(fp.split(dir_prefix)[-1].split(suffix)[0]))
    latest_file = max(file_paths, key=get_iter)
    return latest_file
  except ValueError as verror:
    raise FileNotFoundError(
        f'Files found with pattern \'{search_pattern}\' do not match '
        f'the pattern \'{dir_prefix}[iteration_number]{suffix}\'.\n\n'
        f'Files found:\n{file_paths}') from verror


def get_latest_checkpoint(checkpoint_path):
  """Helper function to get path to latest checkpoint.

  Args:
    checkpoint_path: Path to the directory containing model checkpoints, or
      to a specific checkpoint (e.g. `/path/to/model.ckpt-iteration`).

  Returns:
    Path to latest checkpoint.

  Raises:
    FileNotFoundError: If no checkpoint is found.
  """
  checkpoint_path = expand_path(checkpoint_path)
  is_checkpoint = tf.io.gfile.exists(checkpoint_path + '.index')
  if is_checkpoint:
    # Return the path if it points to a checkpoint.
    return checkpoint_path
  else:
    # Search using 'checkpoints' file.
    # Returns None if no 'checkpoints' file, or directory doesn't exist.
    ckpt = tf.train.latest_checkpoint(checkpoint_path)
    if ckpt:
      return ckpt
    else:
      # Last resort, look for '/path/ckpt-[iter].index' files.
      ckpt_f = get_latest_file(checkpoint_path, prefix='ckpt-', suffix='.index')
      return ckpt_f.split('.index')[0]


# ---------------------------------- Gin ---------------------------------------
def get_latest_operative_config(restore_dir):
  """Finds the most recently saved operative_config in a directory.

  Args:
    restore_dir: Path to directory with gin operative_configs. Will also work
      if passing a path to a file in that directory such as a checkpoint.

  Returns:
    Filepath to most recent operative config.

  Raises:
    FileNotFoundError: If no config is found.
  """
  try:
    return get_latest_file(
        restore_dir, prefix='operative_config-', suffix='.gin')
  except FileNotFoundError:
    return get_latest_file(
        os.path.dirname(restore_dir), prefix='operative_config-', suffix='.gin')


def write_gin_config(summary_writer, save_dir, step):
  """"Writes gin operative_config to save_dir and tensorboard."""
  config_str = gin.operative_config_str()

  # Save the original config string to a file.
  base_name = 'operative_config-{}'.format(step)
  fname = os.path.join(save_dir, base_name + '.gin')
  with tf.io.gfile.GFile(fname, 'w') as f:
    f.write(config_str)

  # Formatting hack copied from gin.tf.GinConfigSaverHook.
  def format_for_tensorboard(line):
    """Convert a single line to markdown format."""
    if not line.startswith('#'):
      return '    ' + line
    line = line[2:]
    if line.startswith('===='):
      return ''
    if line.startswith('None'):
      return '    # None.'
    if line.endswith(':'):
      return '#### ' + line
    return line

  # Convert config string to markdown.
  md_lines = []
  for line in config_str.splitlines():
    md_line = format_for_tensorboard(line)
    if md_line is not None:
      md_lines.append(md_line)
  md_config_str = '\n'.join(md_lines)

  # Add to tensorboard.
  with summary_writer.as_default():
    text_tensor = tf.convert_to_tensor(md_config_str)
    tf.summary.text(name='gin/' + base_name, data=text_tensor, step=step)
    summary_writer.flush()


# ------------------------ Training Loop ---------------------------------------
@gin.configurable
def train(data_provider,
          trainer,
          batch_size=32,
          num_steps=1000000,
          steps_per_summary=300,
          steps_per_save=300,
          save_dir='/tmp/ddsp',
          restore_dir='/tmp/ddsp',
          early_stop_loss_value=None,
          report_loss_to_hypertune=False):
  """Main training loop.

  Args:
   data_provider: DataProvider object for training data.
   trainer: Trainer object built with Model to train.
   batch_size: Total batch size.
   num_steps: Number of training steps.
   steps_per_summary: Number of training steps per summary save.
   steps_per_save: Number of training steps per checkpoint save.
   save_dir: Directory where checkpoints and summaries will be saved.
     If empty string, no checkpoints or summaries will be saved.
   restore_dir: Directory where latest checkpoints for resuming the training
     are stored. If there are no checkpoints in this directory, training will
     begin anew.
   early_stop_loss_value: Early stopping. When the total_loss reaches below this
     value training stops. If None training will run for num_steps steps.
   report_loss_to_hypertune: Report loss values to hypertune package for
     hyperparameter tuning, such as on Google Cloud AI-Platform.
  """
  # Get a distributed dataset iterator.
  dataset = data_provider.get_batch(batch_size, shuffle=True, repeats=-1)
  dataset = trainer.distribute_dataset(dataset)
  dataset_iter = iter(dataset)

  # Build model, easiest to just run forward pass.
  trainer.build(next(dataset_iter))

  # Load latest checkpoint if one exists in load directory.
  try:
    trainer.restore(restore_dir)
  except FileNotFoundError:
    logging.info('No existing checkpoint found in %s, skipping '
                 'checkpoint loading.', restore_dir)

  if save_dir:
    # Set up the summary writer and metrics.
    summary_dir = os.path.join(save_dir, 'summaries', 'train')
    summary_writer = tf.summary.create_file_writer(summary_dir)

    # Save the gin config.
    write_gin_config(summary_writer, save_dir, trainer.step.numpy())
  else:
    # Need to create a dummy writer, even if no save_dir is provided.
    summary_writer = tf.summary.create_noop_writer()

  # Train.
  with summary_writer.as_default():
    tick = time.time()

    first_step = True

    while trainer.step < num_steps:
      step = trainer.step

      # Take a step.
      losses = trainer.train_step(dataset_iter)

      # Create training loss metrics when starting/restarting training.
      if first_step:
        loss_names = list(losses.keys())
        logging.info('Creating metrics for %s', loss_names)
        avg_losses = {name: tf.keras.metrics.Mean(name=name, dtype=tf.float32)
                      for name in loss_names}
        first_step = False

      # Update metrics.
      for k, v in losses.items():
        avg_losses[k].update_state(v)

      # Log the step.
      log_str = 'step: {}\t'.format(int(step.numpy()))
      for k, v in losses.items():
        log_str += '{}: {:.2f}\t'.format(k, v)
      logging.info(log_str)

      # Write Summaries.
      if step % steps_per_summary == 0 and save_dir:
        # Speed.
        steps_per_sec = steps_per_summary / (time.time() - tick)
        tf.summary.scalar('steps_per_sec', steps_per_sec, step=step)
        tick = time.time()

        # Metrics.
        for k, metric in avg_losses.items():
          tf.summary.scalar('losses/{}'.format(k), metric.result(), step=step)
          metric.reset_states()

      # Report metrics for hyperparameter tuning if enabled.
      if report_loss_to_hypertune:
        cloud.report_metric_to_hypertune(losses['total_loss'], step.numpy())

      # Stop the training when the loss reaches given value
      if (early_stop_loss_value is not None and
          losses['total_loss'] <= early_stop_loss_value):
        logging.info('Total loss reached early stopping value of %s',
                     early_stop_loss_value)
        break

      # Save Model.
      if step % steps_per_save == 0 and save_dir:
        trainer.save(save_dir)
        summary_writer.flush()

  # Write a final checkpoint.
  if save_dir:
    trainer.save(save_dir)
    summary_writer.flush()

  logging.info('Training Finished!')
