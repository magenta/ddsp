# Copyright 2021 The DDSP Authors.
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
"""Model base class."""

import time

from absl import logging
import ddsp
from ddsp.core import copy_if_tf_function
from ddsp.training import train_util
import tensorflow as tf


class Model(tf.keras.Model):
  """Wrap the model function for dependency injection with gin."""

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
      self._losses_dict['total_loss'] = tf.reduce_sum(
          list(self._losses_dict.values()))
      return outputs, self._losses_dict

  def _update_losses_dict(self, loss_objs, *args, **kwargs):
    """Helper function to run loss objects on args and add to model losses."""
    for loss_obj in ddsp.core.make_iterable(loss_objs):
      if hasattr(loss_obj, 'get_losses_dict'):
        losses_dict = loss_obj.get_losses_dict(*args, **kwargs)
        self._losses_dict.update(losses_dict)

  def restore(self, checkpoint_path):
    """Restore model and optimizer from a checkpoint."""
    start_time = time.time()
    latest_checkpoint = train_util.get_latest_chekpoint(checkpoint_path)
    if latest_checkpoint is not None:
      checkpoint = tf.train.Checkpoint(model=self)
      checkpoint.restore(latest_checkpoint).expect_partial()
      logging.info('Loaded checkpoint %s', latest_checkpoint)
      logging.info('Loading model took %.1f seconds', time.time() - start_time)
    else:
      logging.info('Could not find checkpoint to load at %s, skipping.',
                   checkpoint_path)

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
