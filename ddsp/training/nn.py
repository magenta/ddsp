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
"""Library of neural network functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow.compat.v1 as tf

tfkl = tf.keras.layers


# ------------------ Normalization ---------------------------------------------
def normalize(x, norm_type='instance', eps=1e-5):
  """Apply either Group, Instance, or Layer normalization."""
  mb, h, w, ch = x.shape
  n_groups = {'instance': ch, 'layer': 1, 'group': 32}[norm_type]

  x = tf.reshape(x, [mb, h, w, n_groups, ch // n_groups])
  mean, var = tf.nn.moments(x, [1, 2, 4], keepdims=True)
  x = (x - mean) / tf.sqrt(var + eps)
  x = tf.reshape(x, [mb, h, w, ch])
  return x


class Normalize(tfkl.Layer):
  """Normalization layer with learnable parameters."""

  def __init__(self, norm_type='instance'):
    super(Normalize, self).__init__()
    self.norm_type = norm_type

  def build(self, x_shape):
    self.scale = self.add_weight('scale',
                                 shape=[1, 1, 1, int(x_shape[-1])],
                                 dtype=tf.float32,
                                 initializer=tf.ones_initializer)
    self.shift = self.add_weight('shift',
                                 shape=[1, 1, 1, int(x_shape[-1])],
                                 dtype=tf.float32,
                                 initializer=tf.zeros_initializer)

  def call(self, x):
    x = normalize(x, self.norm_type)
    return (x * self.scale) + self.shift


# ------------------ ResNet ----------------------------------------------------
def residual_block(x, ch, stride, projection_shortcut, norm_type):
  """A single block for ResNet, with a bottleneck."""
  conv = lambda x, ch, k, s: tfkl.Conv2D(ch, k, s, padding='same')(x)

  r = x
  x = Normalize(norm_type)(x)
  x = tf.nn.relu(x)

  # The projection shortcut should come after the first norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut:
    r = conv(x, 4 * ch, (1, 1), (1, stride))

  x = conv(x, ch, (1, 1), (1, 1))

  x = Normalize(norm_type)(x)
  x = tf.nn.relu(x)
  x = conv(x, ch, (3, 3), (1, stride))

  x = Normalize(norm_type)(x)
  x = tf.nn.relu(x)
  x = conv(x, 4 * ch, (1, 1), (1, 1))
  return x + r


def residual_stack(x, filters, blocks, strides, norm_type):
  """ResNet blocks."""
  for (ch, num_blocks, stride) in zip(filters, blocks, strides):
    # Only the first block per residual_stack uses shortcut and strides
    x = residual_block(x, ch, stride, True, norm_type)

    # Add the additional num_blocks - 1 blocks to the stack.
    for _ in range(1, num_blocks):
      x = residual_block(x, ch, 1, False, norm_type)

  x = Normalize(norm_type)(x)
  x = tf.nn.relu(x)
  return x


@gin.configurable
def resnet(x, norm_type='layer', size='large'):
  """Residual network."""
  ch, blocks = {
      'small': (32, [2, 3, 4]),
      'medium': (32, [3, 4, 6]),
      'large': (64, [3, 4, 6]),
  }[size]
  x = tfkl.Conv2D(64, (7, 7), (1, 2), padding='same')(x)
  x = tf.layers.max_pooling2d(x, pool_size=(1, 3), strides=(1, 2),
                              padding='SAME')

  x = residual_stack(x, [ch, 2*ch, 4*ch], blocks, [1, 2, 2], norm_type)
  x = residual_stack(x, [8*ch], [3], [2], norm_type)
  return x


# ------------------ Stacks ----------------------------------------------------
def fc_stack(x, ch=256, layers=2):
  for _ in range(layers):
    x = tfkl.Dense(ch)(x)
    x = tfkl.LayerNormalization()(x)
    x = tf.nn.leaky_relu(x)
  return x
