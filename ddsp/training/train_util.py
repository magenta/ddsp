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
"""Library of training functions."""

import os
import time

from absl import logging
import gin
import tensorflow.compat.v2 as tf


# ---------------------- Helper Functions --------------------------------------
def get_strategy(tpu='', gpus=None):
  """Create a distribution strategy.

  Args:
    tpu: Address of the TPU. No TPU if left blank.
    gpus: List of GPU addresses for synchronous training.

  Returns:
    A distribution strategy.
  """
  # Get a distribution strategy.
  if tpu:
    logging.info('Use TPU at %s', tpu)
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
  elif gpus:
    for gpu_address in gpus:
      logging.info('Use GPU at %s', gpu_address)
    cluster_spec = tf.train.ClusterSpec({'worker': gpus})
    resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
        cluster_spec=cluster_spec,
        master=gpus[0],
        environment='google',
        rpc_layer='grpc')
    tf.config.experimental_connect_to_cluster(resolver)
    devices = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(devices=devices)
  else:
    logging.info('Defaulting to MirroredStrategy')
    strategy = tf.distribute.MirroredStrategy()
  return strategy


def get_latest_chekpoint(checkpoint_path):
  """Helper function to get path to latest checkpoint.

  Args:
    checkpoint_path: Path to the directory containing model checkpoints, or
      to a specific checkpoint (e.g. `path/to/model.ckpt-iteration`).

  Returns:
    Path to latest checkpoint, or None if none exist.
  """
  checkpoint_path = os.path.expanduser(os.path.expandvars(checkpoint_path))
  is_checkpoint = tf.io.gfile.exists(checkpoint_path + '.index')
  if is_checkpoint:
    return checkpoint_path
  else:
    # None if no checkpoints, or directory doesn't exist.
    return tf.train.latest_checkpoint(checkpoint_path)


def get_latest_operative_config(restore_dir):
  """Finds the most recently saved operative_config in a directory."""
  file_paths = tf.io.gfile.glob(os.path.join(restore_dir, 'operative_config*'))
  get_iter = lambda file_path: int(file_path.split('-')[-1].split('.gin')[0])
  return max(file_paths, key=get_iter) if file_paths else ''


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
          save_dir='~/tmp/ddsp',
          restore_dir='~/tmp/ddsp'):
  """Main training loop."""
  # Get a distributed dataset iterator.
  dataset = data_provider.get_batch(batch_size, shuffle=True, repeats=-1)
  dataset = trainer.distribute_dataset(dataset)
  dataset_iter = iter(dataset)

  # Build model, easiest to just run forward pass.
  trainer.build(next(dataset_iter))

  # Load latest checkpoint if one exists in load directory.
  trainer.restore(restore_dir)

  # Set up the summary writer and metrics.
  summary_dir = os.path.join(save_dir, 'summaries', 'train')
  summary_writer = tf.summary.create_file_writer(summary_dir)

  # Save the gin config.
  write_gin_config(summary_writer, save_dir, trainer.step.numpy())

  # Train.
  with summary_writer.as_default():
    tick = time.time()

    for iteration in range(num_steps):
      step = trainer.step  # Step is not iteration if restarting a model.

      # Take a step.
      losses = trainer.train_step(dataset_iter)

      # Create training loss metrics when starting/restarting training.
      if iteration == 0:
        loss_names = list(losses.keys())
        logging.info('Creating metrics for %s', loss_names)
        avg_losses = {name: tf.keras.metrics.Mean(name=name, dtype=tf.float32)
                      for name in loss_names}

      # Update metrics.
      for k, v in losses.items():
        avg_losses[k].update_state(v)

      # Log the step.
      log_str = 'step: {}\t'.format(int(step.numpy()))
      for k, v in losses.items():
        log_str += '{}: {:.2f}\t'.format(k, v)
      logging.info(log_str)

      # Write Summaries.
      if step % steps_per_summary == 0:
        # Speed.
        steps_per_sec = steps_per_summary / (time.time() - tick)
        tf.summary.scalar('steps_per_sec', steps_per_sec, step=step)
        tick = time.time()

        # Metrics.
        for k, metric in avg_losses.items():
          tf.summary.scalar('losses/{}'.format(k), metric.result(), step=step)
          metric.reset_states()

      # Save Model.
      if step % steps_per_save == 0:
        trainer.save(save_dir)
        summary_writer.flush()

  logging.info('Training Finished!')
