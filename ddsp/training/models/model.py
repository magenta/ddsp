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

"""Model base class."""

import time

from absl import logging
import ddsp
from ddsp.core import copy_if_tf_function
from ddsp.training import train_util
import tensorflow as tf


class Model(tf.keras.Model):
  """Base class for all models."""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._losses_dict = {}

  def __call__(self, *args, return_losses=False, **kwargs):
    """Reset the losses dict on each call.

    Args:
      *args: Arguments passed on to call().
      return_losses: Return a dictionary of losses in addition to the call()
        function returns.
      **kwargs: Keyword arguments passed on to call().

    Returns:
      outputs: A dictionary of model outputs generated in call().
        {output_name: output_tensor or dict}.
      losses: If return_losses=True, also returns a dictionary of losses,
        {loss_name: loss_value}.
    """
    # Copy mutable dicts if in graph mode to prevent side-effects (pure func).
    args = [copy_if_tf_function(a) if isinstance(a, dict) else a for a in args]

    # Run model.
    self._losses_dict = {}
    outputs = super().__call__(*args, **kwargs)

    # Get total loss.
    if not return_losses:
      return outputs
    else:
      self._losses_dict['total_loss'] = self.sum_losses(self._losses_dict)
      return outputs, self._losses_dict

  def sum_losses(self, losses_dict):
    """Sum all the scalar losses in a dictionary."""
    return tf.reduce_sum(list(losses_dict.values()))

  def _update_losses_dict(self, loss_objs, *args, **kwargs):
    """Helper function to run loss objects on args and add to model losses."""
    for loss_obj in ddsp.core.make_iterable(loss_objs):
      if hasattr(loss_obj, 'get_losses_dict'):
        losses_dict = loss_obj.get_losses_dict(*args, **kwargs)
        self._losses_dict.update(losses_dict)

  def restore(self, checkpoint_path, verbose=True, restore_keys=None):
    """Restore model and optimizer from a checkpoint.

    Args:
      checkpoint_path: Path to checkpoint file or directory.
      verbose: Warn about missing variables.
      restore_keys: Optional list of strings for submodules to restore.

    Raises:
      FileNotFoundError: If no checkpoint is found.
    """
    start_time = time.time()
    latest_checkpoint = train_util.get_latest_checkpoint(checkpoint_path)

    if restore_keys is None:
      # If no keys are passed, restore the whole model.
      checkpoint = tf.train.Checkpoint(model=self)
      logging.info('Model restoring all components.')
      if verbose:
        checkpoint.restore(latest_checkpoint)
      else:
        checkpoint.restore(latest_checkpoint).expect_partial()

    else:
      # Restore only sub-modules by building a new subgraph.
      # Following https://www.tensorflow.org/guide/checkpoint#loading_mechanics.
      logging.info('Trainer restoring model subcomponents:')
      for k in restore_keys:
        to_restore = {k: getattr(self, k)}
        log_str = 'Restoring {}'.format(to_restore)
        logging.info(log_str)
        fake_model = tf.train.Checkpoint(**to_restore)
        new_root = tf.train.Checkpoint(model=fake_model)
        status = new_root.restore(latest_checkpoint)
        status.assert_existing_objects_matched()

    logging.info('Loaded checkpoint %s', latest_checkpoint)
    logging.info('Loading model took %.1f seconds', time.time() - start_time)

  def get_audio_from_outputs(self, outputs):
    """Extract audio output tensor from outputs dict of call()."""
    raise NotImplementedError('Must implement `self.get_audio_from_outputs()`.')

  def call(self, *args, training=False, **kwargs):
    """Run the forward pass, add losses, and create a dictionary of outputs.

    This function must run the forward pass, add losses to self._losses_dict and
    return a dictionary of all the relevant output tensors.

    Args:
      *args: Args for forward pass.
      training: Required `training` kwarg passed in by keras.
      **kwargs: kwargs for forward pass.

    Returns:
      Dictionary of all relevant tensors.
    """
    raise NotImplementedError('Must implement a `self.call()` method.')


