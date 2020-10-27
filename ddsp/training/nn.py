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
"""Library of neural network functions."""

import gin
import tensorflow.compat.v2 as tf

tfk = tf.keras
tfkl = tfk.layers


# ------------------------ Shapes ----------------------------------------------
def ensure_4d(x):
  """Add extra dimensions to make sure tensor has height and width."""
  if len(x.shape) == 2:
    return x[:, tf.newaxis, tf.newaxis, :]
  elif len(x.shape) == 3:
    return x[:, :, tf.newaxis, :]
  else:
    return x


def inv_ensure_4d(x, n_dims):
  """Remove excess dims, inverse of ensure_4d() function."""
  if n_dims == 2:
    return x[:, 0, 0, :]
  if n_dims == 3:
    return x[:, :, 0, :]
  else:
    return x


# ------------------ Utilities -------------------------------------------------
@gin.register
def split_to_dict(tensor, tensor_splits):
  """Split a tensor into a dictionary of multiple tensors."""
  labels = [v[0] for v in tensor_splits]
  sizes = [v[1] for v in tensor_splits]
  tensors = tf.split(tensor, sizes, axis=-1)
  return dict(zip(labels, tensors))


# ------------------ Normalization ---------------------------------------------
def normalize_op(x, norm_type='layer', eps=1e-5):
  """Apply either Group, Instance, or Layer normalization, or None."""
  if norm_type is not None:
    mb, h, w, ch = x.shape
    n_groups = {'instance': ch, 'layer': 1, 'group': 32}[norm_type]

    x = tf.reshape(x, [mb, h, w, n_groups, ch // n_groups])
    mean, var = tf.nn.moments(x, [1, 2, 4], keepdims=True)
    x = (x - mean) / tf.sqrt(var + eps)
    x = tf.reshape(x, [mb, h, w, ch])
  return x


@gin.register
class Normalize(tfkl.Layer):
  """Normalization layer with learnable parameters."""

  def __init__(self, norm_type='layer'):
    super().__init__()
    self.norm_type = norm_type

  def build(self, x_shape):
    self.scale = self.add_weight(
        name='scale',
        shape=[1, 1, 1, int(x_shape[-1])],
        dtype=tf.float32,
        initializer=tf.ones_initializer)
    self.shift = self.add_weight(
        name='shift',
        shape=[1, 1, 1, int(x_shape[-1])],
        dtype=tf.float32,
        initializer=tf.zeros_initializer)

  def call(self, x):
    n_dims = len(x.shape)
    x = ensure_4d(x)
    x = normalize_op(x, self.norm_type)
    x = (x * self.scale) + self.shift
    return inv_ensure_4d(x, n_dims)


# ------------------ ResNet ----------------------------------------------------
@gin.register
class NormReluConv(tf.keras.Sequential):
  """Norm -> ReLU -> Conv layer."""

  def __init__(self, ch, k, s, norm_type, **kwargs):
    """Downsample frequency by stride."""
    layers = [
        Normalize(norm_type),
        tfkl.Activation(tf.nn.relu),
        tfkl.Conv2D(ch, (k, k), (1, s), padding='same'),
    ]
    super().__init__(layers, **kwargs)


@gin.register
class ResidualLayer(tfkl.Layer):
  """A single layer for ResNet, with a bottleneck."""

  def __init__(self, ch, stride, shortcut, norm_type, **kwargs):
    """Downsample frequency by stride, upsample channels by 4."""
    super().__init__(**kwargs)
    ch_out = 4 * ch
    self.shortcut = shortcut

    # Layers.
    self.norm_input = Normalize(norm_type)
    if self.shortcut:
      self.conv_proj = tfkl.Conv2D(
          ch_out, (1, 1), (1, stride), padding='same', name='conv_proj')
    layers = [
        tfkl.Conv2D(ch, (1, 1), (1, 1), padding='same'),
        NormReluConv(ch, 3, stride, norm_type),
        NormReluConv(ch_out, 1, 1, norm_type),
    ]
    self.bottleneck = tf.keras.Sequential(layers, name='bottleneck')

  def call(self, x):
    r = x
    x = tf.nn.relu(self.norm_input(x))
    # The projection shortcut should come after the first norm and ReLU
    # since it performs a 1x1 convolution.
    r = self.conv_proj(x) if self.shortcut else r
    x = self.bottleneck(x)
    return x + r


@gin.register
class ResidualStack(tf.keras.Sequential):
  """LayerNorm -> ReLU -> Conv layer."""

  def __init__(self,
               filters,
               block_sizes,
               strides,
               norm_type,
               **kwargs):
    """ResNet layers."""
    layers = []
    for (ch, n_layers, stride) in zip(filters, block_sizes, strides):
      # Only the first block per residual_stack uses shortcut and strides.
      layers.append(ResidualLayer(ch, stride, True, norm_type))
      # Add the additional (n_layers - 1) layers to the stack.
      for _ in range(1, n_layers):
        layers.append(ResidualLayer(ch, 1, False, norm_type))
    layers.append(Normalize(norm_type))
    layers.append(tfkl.Activation(tf.nn.relu))
    super().__init__(layers, **kwargs)


@gin.register
class ResNet(tf.keras.Sequential):
  """Residual network."""

  def __init__(self, size='large', norm_type='layer', **kwargs):
    size_dict = {
        'small': (32, [2, 3, 4]),
        'medium': (32, [3, 4, 6]),
        'large': (64, [3, 4, 6]),
    }
    ch, blocks = size_dict[size]
    layers = [
        tfkl.Conv2D(64, (7, 7), (1, 2), padding='same'),
        tfkl.MaxPool2D(pool_size=(1, 3), strides=(1, 2), padding='same'),
        ResidualStack([ch, 2 * ch, 4 * ch], blocks, [1, 2, 2], norm_type),
        ResidualStack([8 * ch], [3], [2], norm_type)
    ]
    super().__init__(layers, **kwargs)


# ---------------- Stacks ------------------------------------------------------
@gin.register
class Fc(tf.keras.Sequential):
  """Makes a Dense -> LayerNorm -> Leaky ReLU layer."""

  def __init__(self, ch=128, **kwargs):
    layers = [
        tfkl.Dense(ch),
        tfkl.LayerNormalization(),
        tfkl.Activation(tf.nn.leaky_relu),
    ]
    super().__init__(layers, **kwargs)


@gin.register
class FcStack(tf.keras.Sequential):
  """Stack Dense -> LayerNorm -> Leaky ReLU layers."""

  def __init__(self, ch=256, layers=2, **kwargs):
    layers = [Fc(ch) for i in range(layers)]
    super().__init__(layers, **kwargs)


@gin.register
class Rnn(tfkl.Layer):
  """Single RNN layer."""

  def __init__(self, dims, rnn_type, return_sequences=True, **kwargs):
    super().__init__(**kwargs)
    rnn_class = {'lstm': tfkl.LSTM, 'gru': tfkl.GRU}[rnn_type]
    self.rnn = rnn_class(dims, return_sequences=return_sequences)

  def call(self, x):
    return self.rnn(x)


