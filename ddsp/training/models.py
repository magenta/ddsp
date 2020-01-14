# Copyright 2019 The DDSP Authors.
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
"""Model that outputs coefficeints of an additive synthesizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import logging

import ddsp
from ddsp.training import train_util

import gin
import tensorflow.compat.v1 as tf

tfkl = tf.keras.layers


@gin.configurable
def get_model(model=gin.REQUIRED):
  """Gin configurable function get a 'global' model for use in ddsp_run.py.

  Convenience for using the same model in train(), evaluate(), and sample().
  Args:
    model: An instantiated model, such as 'models.Autoencoder()'.
  Returns:
    The 'global' model specifieed in the gin config.
  """
  return model


class Model(tfkl.Layer):
  """Wrap the model function for dependency injection with gin."""

  def __init__(self, name='model'):
    super(Model, self).__init__(name=name)
    self.tb_metrics = {}

  def call(self, features, training=True):
    return self.get_outputs(features, training=training)

  def get_outputs(self, features, training=True):
    """Return a dictionary of the all relevant tensors for model inspection.

    Args:
      features: An input dictionary of data feature tensors.
      training: Different behavior for training.

    Returns:
      Output dictionary of tensors. Must include a scalar tensor for the key
        'total_loss'.
    """
    raise NotImplementedError

  def get_scaffold_fn(self):
    """Optionally returns scaffold_fn."""
    return

  def get_variables_to_optimize(self):
    """Returns variables to optimize."""
    return tf.trainable_variables()

  def add_tb_metric(self, name, tensor):
    """Add scalar average metric to be plotted on tensorboard."""
    metric_tensor = tf.reduce_mean(tensor)
    metric_tensor = tf.expand_dims(metric_tensor, axis=0)
    self.tb_metrics[name] = metric_tensor

  def get_model_fn(self, use_tpu=True):
    """Returns function for Estimator."""

    def model_fn(features, labels, mode, params, config):
      """Builds the network model."""
      del labels
      del config
      outputs = self.get_outputs(features)
      model_dir = params['model_dir']

      host_call = (train_util.get_host_call_fn(model_dir), self.tb_metrics)

      estimator_spec = train_util.get_estimator_spec(
          outputs['total_loss'],
          mode,
          model_dir,
          use_tpu=use_tpu,
          scaffold_fn=self.get_scaffold_fn(),
          variables_to_optimize=self.get_variables_to_optimize(),
          host_call=host_call)
      return estimator_spec

    return model_fn

  def restore(self, sess, checkpoint_path):
    """Load weights from most recent checkpoint in directory.

    Args:
      sess: tf.Session() with which to load the checkpoint
      checkpoint_path: Path to the directory containing model checkpoints, or to
        a specific checkpoint. For example, `path/to/model.ckpt-iteration`.
    """
    start_time = time.time()
    trainable_variables = self.get_variables_to_optimize()
    saver = tf.train.Saver(var_list=trainable_variables)

    checkpoint_path = os.path.expanduser(os.path.expandvars(checkpoint_path))
    if tf.gfile.IsDirectory(checkpoint_path):
      checkpoint = tf.train.latest_checkpoint(checkpoint_path)
    else:
      checkpoint = checkpoint_path
    saver.restore(sess, checkpoint)
    logging.info('Loading model took %.1f seconds', time.time() - start_time)


@gin.configurable
class Autoencoder(Model):
  """Wrap the model function for dependency injection with gin."""

  def __init__(self,
               preprocessor=None,
               encoder=None,
               decoder=None,
               processor_group=None,
               losses=None,
               name='autoencoder'):
    super(Autoencoder, self).__init__(name=name)
    self.preprocessor = preprocessor
    self.encoder = encoder
    self.decoder = decoder
    self.processor_group = processor_group
    self.loss_objs = ddsp.core.make_iterable(losses)

  def get_outputs(self, features, training=True):
    """Run the core of the network, get predictions and loss.

    Args:
      features: An input dictionary of audio features. Requires at least the
        item "audio" (tensor[batch, n_samples]).
      training: Different behavior for training.

    Returns:
      Dictionary of all inputs, decoder outputs, signal processor intermediates,
        and losses. Includes total loss and audio_gen.
    """
    # ---------------------- Data Preprocessing --------------------------------
    # Returns a modified copy of features.
    conditioning = self.preprocessor(features, training=training)

    # ---------------------- Encoder -------------------------------------------
    if self.encoder is not None:
      conditioning = self.encoder(conditioning)

    # ---------------------- Decoder -------------------------------------------
    conditioning = self.decoder(conditioning)

    # ---------------------- Synthesizer ---------------------------------------
    outputs = self.processor_group.get_outputs(conditioning)
    audio_gen = outputs[self.processor_group.name]['signal']
    outputs['audio_gen'] = audio_gen

    # ---------------------- Losses --------------------------------------------
    total_loss = 0.0
    loss_dict = {}
    for loss_obj in self.loss_objs:
      loss_term = loss_obj(features['audio'], audio_gen)
      total_loss += loss_term
      loss_name = 'losses/{}'.format(loss_obj.name)
      loss_dict[loss_name] = loss_term
      self.add_tb_metric(loss_name, loss_term)

    # Update tb and outputs.
    self.add_tb_metric('loss', total_loss)
    self.add_tb_metric('global_step', tf.train.get_or_create_global_step())
    outputs.update(loss_dict)
    outputs['total_loss'] = total_loss

    return outputs

  @property
  def pretrained_models(self):
    pretrained_models = []
    for loss_obj in self.loss_objs:
      m = loss_obj.pretrained_model
      if m is not None:
        pretrained_models.append(m)
    return pretrained_models

  def get_scaffold_fn(self):
    """Returns scaffold_fn."""

    def scaffold_fn():
      """scaffold_fn."""
      # load pretrained model weights
      for pretrained_model in self.pretrained_models:
        pretrained_model.init_from_checkpoint()

      return tf.train.Scaffold()

    return scaffold_fn

  def get_variables_to_optimize(self):
    """Returns variables to optimize."""
    all_trainables = tf.trainable_variables()
    vars_to_freeze = []
    for m in self.pretrained_models:
      vars_to_freeze += m.trainable_variables()
    var_names_to_freeze = [x.name for x in vars_to_freeze]

    trainables = []
    for x in all_trainables:
      if x.name not in var_names_to_freeze:
        trainables.append(x)
        logging.info('adding trainable variable %s (shape=%s, dtype=%s).',
                     x.name, x.shape, x.dtype)
      else:
        logging.info('!skipping frozen variable %s (shape=%s, dtype=%s).',
                     x.name, x.shape, x.dtype)
    return trainables
