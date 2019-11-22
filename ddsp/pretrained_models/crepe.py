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
"""Pretrained CREPE model.

See gin/experiments/additive/pretrained.gin for how to configure the CREPE model
hyper-parameters.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from ddsp.pretrained_models import base
import gin
import gin.tf
import tensorflow.compat.v1 as tf


class Crepe(tf.keras.layers.Layer):
  """Crepe model.

    Paper: https://arxiv.org/pdf/1802.06182.pdf

    Adapted from: https://github.com/marl/crepe/blob/master/crepe/core.py
  """

  def __init__(self, model_capacity, activation_layer, name=None):
    super(Crepe, self).__init__(self, name=name)
    self._model_capacity = model_capacity
    self._capacity_multiplier = {
        'tiny': 4,
        'small': 8,
        'medium': 16,
        'large': 24,
        'full': 32
    }[model_capacity]
    self._activation_layer = activation_layer
    self._layers = self._build_layers()

  def _build_layers(self):
    """Returns layers in the CREPE model."""
    layers = []
    k = tf.keras.layers  # short-cut for readability

    layer_ids = [1, 2, 3, 4, 5, 6]
    filters = [n * self._capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
    widths = [512, 64, 64, 64, 64, 64]
    strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

    for l, f, w, s in zip(layer_ids, filters, widths, strides):
      layers.append(
          k.Conv2D(
              f, (w, 1),
              strides=s,
              padding='same',
              activation='relu',
              name='conv%d' % l))

      layers.append(k.BatchNormalization(name='conv%d-BN' % l))
      layers.append(
          k.MaxPool2D(
              pool_size=(2, 1),
              strides=None,
              padding='valid',
              name='conv%d-maxpool' % l))
      layers.append(k.Dropout(0.25, name='conv%d-dropout' % l))

    layers.append(k.Permute((2, 1, 3), name='transpose'))
    layers.append(k.Flatten(name='flatten'))
    layers.append(k.Dense(360, activation='sigmoid', name='classifier'))

    return layers

  def call(self, features, training=False):
    # returns a dict of tensors, from layer name to layer activations.
    assert features.shape[1] == 1024

    y = features
    for _ in range(2):
      y = tf.expand_dims(y, axis=-1)  # [batch, length, 1, 1]

    activation_dict = {}

    for layer in self._layers:
      if 'BN' in layer.name or 'dropout' in layer.name:
        y = layer(y, training=training)
      else:
        y = layer(y)
      activation_dict[layer.name] = y
    return activation_dict


@gin.configurable
class PretrainedCREPE(base.PretrainedModel):
  """CREPE model."""

  def __init__(self,
               model_capacity='tiny',
               activation_layer='classifier',
               name='crepe',
               checkpoint='/path/to/crepe/model-tiny.ckpt'):
    super(PretrainedCREPE, self).__init__(name=name, checkpoint=checkpoint)
    self._model_capacity = model_capacity
    self._activation_layer = activation_layer
    self._model = None

  def _build_model(self):
    """Builds the CREPE model."""
    self._model = Crepe(
        self._model_capacity, self._activation_layer, name=self._name)

  def __call__(self, features):
    return self.get_outputs(features)

  def get_outputs(self, features):
    """Returns the embeddings.

    Args:
      features: tensors of shape [batch, length]. length must be divisible by
        1024.

    Returns:
      activations of shape [batch, depth]
    """
    if self._model is None:
      self._build_model()
    batch_size = tf.shape(features)[0]
    length = int(features.shape[1])

    # TODO(gcj): relax this constraint by modifying the model to generate
    # outputs at every time point.
    if length % 1024:
      truncated_length = length - length % 1024
      logging.warning(
          ('Length of the tensor must be multiples of 1024, but is %d. '
           'Truncating the input length to %d.'), length, truncated_length)
      features = tf.slice(features, [0, 0], [-1, truncated_length])
    features = tf.reshape(features, [-1, 1024])
    activation_dict = self._model(features)
    if self._activation_layer not in activation_dict:
      raise ValueError(
          'activation layer {} not found, valid keys are {}'.format(
              self._activation_layer, sorted(activation_dict.keys())))
    outputs = activation_dict.get(self._activation_layer)
    outputs = tf.reshape(outputs, [batch_size, -1])
    return outputs
