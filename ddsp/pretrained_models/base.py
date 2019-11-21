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
"""Base class for pretrained models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import tensorflow.compat.v1 as tf


class PretrainedModel(object):
  """Base class that wrap any pretrained model."""

  def __init__(self, name='', checkpoint=''):
    self._name = name
    self._checkpoint = checkpoint

  @property
  def name(self):
    return self._name

  @property
  def variable_scope(self):
    return self._name

  def __call__(self, features):
    with tf.variable_scope(self.variable_scope, reuse=tf.AUTO_REUSE):
      outputs = self.get_outputs(features)
      return outputs

  def get_outputs(self, features):
    """Returns the output of the model, usually an embedding."""
    raise NotImplementedError

  def _get_ckpt_var_name(self, var_name):
    """Derives the variable name in the checkpoint.

    Args:
      var_name: str, the current variable name
    Returns:
      the variable name in the checkpoint.
    """
    # remove prefix
    prefix = self.variable_scope + '/'
    if not var_name.startswith(prefix):
      raise ValueError('variable {} does not start with the scope {}.'.format(
          var_name, prefix))
    ckpt_var_name = var_name[len(prefix):]

    # remove suffix
    if ckpt_var_name.endswith(':0'):
      ckpt_var_name = ckpt_var_name[:-2]

    return ckpt_var_name

  def init_from_checkpoint(self):
    """Replaces the initializer by restore ops."""
    logging.info('loading pretrained checkpoint %s to %s.', self._checkpoint,
                 self.variable_scope)
    # TODO(gcj): add support to load EMAed weights.
    # NOTE: we need to explictly build the assignment_map (instead of only
    # specifying the old scope and the new scope) because tf.Variable() do not
    # belong to any variable scope even though their variable name contains the
    # scope in which the variable is created. tf.Variable() is used in all keras
    # layers.
    assignment_map = {}
    for v in self.trainable_variables():
      ckpt_var_name = self._get_ckpt_var_name(v.name)
      assignment_map[ckpt_var_name] = v
      logging.info('  assignment_map[%s] = %s', ckpt_var_name, v)
    tf.train.init_from_checkpoint(self._checkpoint, assignment_map)

  def trainable_variables(self):
    return tf.trainable_variables(scope=self.variable_scope)
