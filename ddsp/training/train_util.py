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
"""Library of training functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

import gin
import gin.tf
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2.summary as tf_summary


# ---------------------- Train op ----------------------------------------------
def _clip_gradients_by_norm(grads_and_vars, gradient_clip_norm):
  """Clips gradients by global norm."""
  gradients, variables = zip(*grads_and_vars)
  clipped_gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip_norm)
  return list(zip(clipped_gradients, variables))


@gin.configurable
def get_train_op(loss,
                 learning_rate=0.001,
                 lr_decay_steps=10000,
                 lr_decay_rate=0.98,
                 gradient_clip_norm=3.0,
                 use_tpu=True,
                 variables=None):
  """Get training operation with gradient clipping and learning rate decay.

  Distilled from tf.contrib.layers.optimize_loss().
  Args:
    loss: Scalar tensor of the loss function.
    learning_rate: Scalar initial learning rate.
    lr_decay_steps: Exponential decay timescale.
    lr_decay_rate: Exponential decay magnitude.
    gradient_clip_norm: Global norm by which to scale gradients.
    use_tpu: Use tpu for training.
    variables: List of variables to optimize. tf.trainable_variables() if None.

  Returns:
    train_op: Operation that runs one iteration of training.
  """
  global_step = tf.train.get_or_create_global_step()

  with tf.variable_scope('training', values=[loss, global_step]):
    # Make sure update ops run before computing loss.
    update_ops = list(set(tf.get_collection(tf.GraphKeys.UPDATE_OPS)))
    with tf.control_dependencies(update_ops):
      loss = tf.identity(loss)

    # Learning rate variable, with decay.
    learning_rate_decay_fn = functools.partial(
        tf.train.exponential_decay,
        decay_steps=lr_decay_steps,
        decay_rate=lr_decay_rate,
        staircase=True)
    lr = tf.get_variable(
        'learning_rate', [],
        trainable=False,
        initializer=tf.constant_initializer(learning_rate))
    lr = learning_rate_decay_fn(lr, global_step)

    # Optimizer.
    opt = tf.train.AdamOptimizer(lr)
    if use_tpu:
      opt = tf.tpu.CrossShardOptimizer(opt)

    # All trainable variables, if specific variables are not specified.
    if variables is None:
      variables = tf.trainable_variables()

    # Compute gradients.
    gradients = opt.compute_gradients(
        loss, variables, colocate_gradients_with_ops=False)

    # Optionally clip gradients by global norm.
    if isinstance(gradient_clip_norm, float):
      gradients = _clip_gradients_by_norm(gradients, gradient_clip_norm)

    # Create gradient updates.
    grad_updates = opt.apply_gradients(
        gradients, global_step=global_step, name='train')

    # Ensure the train_op computes grad_updates.
    with tf.control_dependencies([grad_updates]):
      train_op = tf.identity(loss)

    return train_op


# ---------------------- Estimators --------------------------------------------
def get_estimator_spec(loss,
                       mode,
                       model_dir,
                       use_tpu=True,
                       scaffold_fn=None,
                       variables_to_optimize=None,
                       host_call=None):
  """Get TPUEstimatorSpec depending on mode, for Model.get_model_fn()."""
  train_op = get_train_op(
      loss, use_tpu=use_tpu, variables=variables_to_optimize)
  gin_config_saver_hook = gin.tf.GinConfigSaverHook(
      model_dir, summarize_config=True)

  # Train
  if mode == tf.estimator.ModeKeys.TRAIN:
    return tf.estimator.tpu.TPUEstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        training_hooks=[
            gin_config_saver_hook,
        ],
        scaffold_fn=scaffold_fn,
        host_call=host_call)

  # Eval
  elif mode == tf.estimator.ModeKeys.EVAL:
    raise ValueError('Estimator evaluation is not supported. Use ddsp_run.py '
                     '--mode=eval instead.')

  # Predict
  elif mode == tf.estimator.ModeKeys.PREDICT:
    raise ValueError('Do not use estimator.predict(), which requires a flat '
                     'dictionary of predictions. Use model.get_outputs() and '
                     'model.restore() instead.')
  else:
    raise ValueError('Unsupported mode: %s' % mode)


def get_host_call_fn(model_dir):
  """`host_call` function for creating training summaries when using TPU."""

  def host_call_fn(**kwargs):
    """Host_call_fn.

    Args:
      **kwargs: dict of summary name to tf.Tensor mapping. The value we see here
        is the tensor across all cores, concatenated along axis 0. This function
        will take make a scalar summary that is the mean of the whole tensor (as
        all the values are the same - the mean, trait of
        tpu.CrossShardOptimizer).

    Returns:
      A merged summary op.
    """
    gs = kwargs.pop('global_step')[0]
    with tf_summary.create_file_writer(model_dir).as_default():
      with tf_summary.record_if(tf.equal(gs % 10, 0)):
        for name, tensor in kwargs.items():
          # Take the mean across cores.
          tensor = tf.reduce_mean(tensor)
          tf_summary.scalar(name, tensor, step=gs)
        return tf.summary.all_v2_summary_ops()

  return host_call_fn


@gin.configurable
def create_estimator(model_fn,
                     model_dir,
                     master='',
                     batch_size=128,
                     save_checkpoint_steps=300,
                     save_summary_steps=300,
                     keep_checkpoint_max=100,
                     warm_start_from=None,
                     use_tpu=True):
  """Creates an estimator."""
  config = tf.estimator.tpu.RunConfig(
      master=master,
      tpu_config=tf.estimator.tpu.TPUConfig(
          iterations_per_loop=save_checkpoint_steps),
      save_summary_steps=save_summary_steps,
      save_checkpoints_steps=save_checkpoint_steps,
      keep_checkpoint_max=keep_checkpoint_max,
      keep_checkpoint_every_n_hours=1)

  params = {'model_dir': model_dir}
  return tf.estimator.tpu.TPUEstimator(
      model_fn=model_fn,
      model_dir=model_dir,
      params=params,
      train_batch_size=batch_size,
      eval_batch_size=batch_size,
      predict_batch_size=batch_size,
      config=config,
      warm_start_from=warm_start_from,
      use_tpu=use_tpu,
      eval_on_tpu=False)


# ---------------------- Training ----------------------------------------------
@gin.configurable
def train(data_provider,
          model,
          model_dir='~/tmp/ddsp',
          num_steps=1000000,
          master='',
          use_tpu=True):
  """Main training loop."""
  input_fn = data_provider.get_input_fn(shuffle=True, repeats=-1)
  model_fn = model.get_model_fn(use_tpu=use_tpu)

  estimator = create_estimator(
      model_fn=model_fn,
      model_dir=os.path.expanduser(model_dir),
      master=master,
      use_tpu=use_tpu)

  estimator.train(input_fn=input_fn, max_steps=num_steps)
