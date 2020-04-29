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


@gin.configurable
class Trainer(object):
  """Class to bind an optimizer, model, strategy, and training step function."""

  def __init__(self,
               model,
               strategy,
               checkpoints_to_keep=100,
               learning_rate=0.001,
               lr_decay_steps=10000,
               lr_decay_rate=0.98,
               grad_clip_norm=3.0,
               restore_keys=None):
    """Constructor.

    Args:
      model: Model to train.
      strategy: A distribution strategy.
      checkpoints_to_keep: Max number of checkpoints before deleting oldest.
      learning_rate: Scalar initial learning rate.
      lr_decay_steps: Exponential decay timescale.
      lr_decay_rate: Exponential decay magnitude.
      grad_clip_norm: Norm level by which to clip gradients.
      restore_keys: List of names of model properties to restore. If no keys are
        passed, restore the whole model.
    """
    self.model = model
    self.strategy = strategy
    self.checkpoints_to_keep = checkpoints_to_keep
    self.grad_clip_norm = grad_clip_norm
    self.restore_keys = restore_keys

    # Create an optimizer.
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=lr_decay_steps,
        decay_rate=lr_decay_rate)

    with self.strategy.scope():
      optimizer = tf.keras.optimizers.Adam(lr_schedule)
      self.optimizer = optimizer

  def save(self, save_dir):
    """Saves model and optimizer to a checkpoint."""
    # Saving weights in checkpoint format because saved_model requires
    # handling variable batch size, which some synths and effects can't.
    start_time = time.time()
    checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
    manager = tf.train.CheckpointManager(
        checkpoint, directory=save_dir, max_to_keep=self.checkpoints_to_keep)
    step = self.step.numpy()
    manager.save(checkpoint_number=step)
    logging.info('Saved checkpoint to %s at step %s', save_dir, step)
    logging.info('Saving model took %.1f seconds', time.time() - start_time)

  def restore(self, checkpoint_path, restore_keys=None):
    """Restore model and optimizer from a checkpoint if it exists."""
    logging.info('Restoring from checkpoint...')
    start_time = time.time()

    # Prefer function args over object properties.
    restore_keys = self.restore_keys if restore_keys is None else restore_keys
    if restore_keys is None:
      # If no keys are passed, restore the whole model.
      model = self.model
      logging.info('Trainer restoring the full model')
    else:
      # Restore only sub-modules by building a new subgraph.
      restore_dict = {k: getattr(self.model, k) for k in restore_keys}
      model = tf.train.Checkpoint(**restore_dict)

      logging.info('Trainer restoring model subcomponents:')
      for k, v in restore_dict.items():
        log_str = 'Restoring {}: {}'.format(k, v)
        logging.info(log_str)

    # Restore from latest checkpoint.
    checkpoint = tf.train.Checkpoint(model=model, optimizer=self.optimizer)
    latest_checkpoint = get_latest_chekpoint(checkpoint_path)
    if latest_checkpoint is not None:
      # checkpoint.restore must be within a strategy.scope() so that optimizer
      # slot variables are mirrored.
      with self.strategy.scope():
        if restore_keys is None:
          checkpoint.restore(latest_checkpoint)
        else:
          checkpoint.restore(latest_checkpoint).expect_partial()
        logging.info('Loaded checkpoint %s', latest_checkpoint)
      logging.info('Loading model took %.1f seconds', time.time() - start_time)
    else:
      logging.info('No checkpoint, skipping.')

  @property
  def step(self):
    """The number of training steps completed."""
    return self.optimizer.iterations

  def psum(self, x, axis=None):
    """Sum across processors."""
    return self.strategy.reduce(tf.distribute.ReduceOp.SUM, x, axis=axis)

  def run(self, fn, *args, **kwargs):
    """Distribute and run function on processors."""
    return self.strategy.experimental_run_v2(fn, args=args, kwargs=kwargs)

  def build(self, batch):
    """Build the model by running a distributed batch through it."""
    logging.info('Building the model...')
    _ = self.run(tf.function(self.model.__call__), batch)
    self.model.summary()

  def distribute_dataset(self, dataset):
    """Create a distributed dataset."""
    if isinstance(dataset, tf.data.Dataset):
      return self.strategy.experimental_distribute_dataset(dataset)
    else:
      return dataset

  @tf.function
  def train_step(self, dataset_iter):
    """Distributed training step."""
    # Wrap in distribution strategy, slight speedup passing in iter vs batch.
    batch = next(dataset_iter)
    losses = self.run(self.step_fn, batch)
    # Add up the scalar losses across replicas.
    n_replicas = self.strategy.num_replicas_in_sync
    return {k: self.psum(v, axis=None) / n_replicas for k, v in losses.items()}

  @tf.function
  def step_fn(self, batch):
    """Per-Replica training step."""
    with tf.GradientTape() as tape:
      _, losses = self.model(batch, return_losses=True, training=True)
    # Clip and apply gradients.
    grads = tape.gradient(losses['total_loss'], self.model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, self.grad_clip_norm)
    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    return losses


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
  # Get a distributed dataset.
  dataset = data_provider.get_batch(batch_size, shuffle=True, repeats=-1)
  dataset = trainer.distribute_dataset(dataset)
  dataset_iter = iter(dataset)

  # Load latest checkpoint if one exists in load directory.
  trainer.restore(restore_dir)

  # Build model, easiest to just run forward pass.
  trainer.build(next(dataset_iter))

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
