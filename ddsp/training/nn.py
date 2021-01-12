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
"""Library of neural network functions."""

import inspect

from ddsp import core
from ddsp import spectral_ops
import gin
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

tfk = tf.keras
tfkl = tfk.layers


class DictLayer(tfkl.Layer):
  """Wrap a Keras Layer to take dictionary inputs and outputs.

  Note that all return values will be converted to a dictionary, even if
  the `call()` returns a tuple. For instance, a function like so:

  ```
    class MyLayer(DictLayer):
      # --- (init ignored)

      def call(self, a, b, c) --> ['x', 'y', 'z']:

        # Do something cool

        return a, b, c  # Note: returned as tuple in call().
  ```

  Will return the following:

  >>> my_layer = MyLayer()
  >>> my_layer(1, 2, 3)
  {'x': 1, 'y': 2, 'z': 3}  # Note: returned values is dict when called.

  """

  def __init__(self, input_keys=None, output_keys=None, **kwargs):
    """Constructor, define input and output keys.

    Args:
      input_keys: A list of keys to read out of a dictionary passed to call().
        If no input_keys are provided to the constructor, they are inferred from
        the argument names in call(). Input_keys are ignored if call() recieves
        tensors as arguments instead of a dict.
      output_keys: A list of keys to name the outputs returned from call(), and
        construct an outputs dictionary. If call() returns a dictionary, these
        keys are ignored. If no output_keys are provided to the constructor,
        they are inferred from return annotation of call() (a list of strings).
      **kwargs: Other keras layer kwargs such as name.
    """
    super().__init__(**kwargs)
    input_keys = input_keys or self.get_argument_names('call')
    output_keys = output_keys or self.get_return_annotations('call')

    self.input_keys = list(input_keys)
    self.output_keys = list(output_keys)
    self.default_input_keys = self.get_default_argument_names('call')

  def __call__(self, *inputs, **kwargs):
    """Wrap the layer's __call__() with dictionary inputs and outputs.

    IMPORTANT: If no input_keys are provided to the constructor, they are
    inferred from the argument names in call(). If no output_keys are provided
    to the constructor, they are inferred from return annotation of call()
    (a list of strings).

    Example:
    ========
    ```
    def call(self, f0_hz, loudness, power=None) -> ['amps', 'frequencies']:
      ...
      return amps, frequencies
    ```
    Will infer `self.input_keys = ['f0_hz', 'loudness']` and
    `self.output_keys = ['amps', 'frequencies']`. If input_keys, or output_keys
    are provided to the constructor they will override these inferred values.
    It will also infer `self.default_input_keys = ['power']`, which it will try
    to look up the inputs, but use the default values and not throw an error if
    the key is not in the input dictionary.

    Example Usage:
    ==============
    The the example above works with both tensor inputs `layer(f0_hz, loudness)`
    or `layer(f0_hz, loudness, power)` or a dictionary of tensors
    `layer({'f0_hz':..., 'loudness':...})`, or
    `layer({'f0_hz':..., 'loudness':..., 'power':...})` and in both cases will
    return a dictionary of tensors `{'amps':..., 'frequencies':...}`.

    Args:
      *inputs: Arguments passed on to call(). If any arguments are dicts, they
        will be merged and self.input_keys will be read out of them and passed
        to call() while other args will be ignored.
      **kwargs: Keyword arguments passed on to call().

    Returns:
      outputs: A dictionary of layer outputs from call(). If the layer call()
        returns a dictionary it will be returned directly, otherwise the output
        tensors will be wrapped in a dictionary {output_key: output_tensor}.
    """
    # Merge all dictionaries provided in inputs.
    input_dict = {}
    for v in inputs:
      if isinstance(v, dict):
        input_dict.update(v)

    # If any dicts provided, lookup input tensors from those dicts.
    # Otherwise, just use inputs list as input tensors.
    if input_dict:
      inputs = [core.nested_lookup(key, input_dict) for key in self.input_keys]
      # Optionally add for default arguments if key is present in input_dict.
      for key in  self.default_input_keys:
        try:
          inputs.append(core.nested_lookup(key, input_dict))
        except KeyError:
          pass

    # Run input tensors through the model.
    outputs = super().__call__(*inputs, **kwargs)

    # Return dict if call() returns it.
    if isinstance(outputs, dict):
      return outputs
    # Otherwise make a dict from output_keys.
    else:
      outputs = core.make_iterable(outputs)
      if len(self.output_keys) != len(outputs):
        raise ValueError(f'Output keys ({self.output_keys}) must have the same'
                         f'length as outputs ({outputs})')
      return dict(zip(self.output_keys, outputs))

  def get_argument_names(self, method):
    """Get list of strings for names of required arguments to method."""
    spec = inspect.getfullargspec(getattr(self, method))
    if spec.defaults:
      n_defaults = len(spec.defaults)
      return spec.args[1:-n_defaults]
    else:
      return spec.args[1:]

  def get_default_argument_names(self, method):
    """Get list of strings for names of default arguments to method."""
    spec = inspect.getfullargspec(getattr(self, method))
    if spec.defaults:
      n_defaults = len(spec.defaults)
      return spec.args[-n_defaults:]
    else:
      return []

  def get_return_annotations(self, method):
    """Get list of strings of return annotations of method."""
    spec = inspect.getfullargspec(getattr(self, method))
    return core.make_iterable(spec.annotations['return'])


class OutputSplitsLayer(DictLayer):
  """A DictLayer that splits an output tensor into a dictionary of tensors."""

  def __init__(self,
               input_keys=None,
               output_splits=(('amps', 1), ('harmonic_distribution', 40)),
               **kwargs):
    """Layer constructor.

    A common architecture is to have a homogenous network with a final dense
    layer for each output type, for instance, for each parameter of a
    synthesizer. This base layer wraps this process by just requiring that
    `compute_output()` return a single tensor, which is then run through a dense
    layer and split into a dict according to `output_splits`.

    Args:
      input_keys: A list of keys to read out of a dictionary passed to call().
        If no input_keys are provided to the constructor, they are inferred from
        the argument names in compute_outputs().
      output_splits: A list of tuples (output_key, n_channels). Output keys are
        extracted from the list and the output tensor from compute_output(), is
        split into a dictionary of tensors, each with its matching n_channels.
      **kwargs: Other tf.keras.layer kwargs, such as name.
    """
    self.output_splits = output_splits
    self.n_out = sum([v[1] for v in output_splits])
    self.dense_out = tfkl.Dense(self.n_out)
    input_keys = input_keys or self.get_argument_names('compute_output')
    output_keys = [v[0] for v in output_splits]
    super().__init__(input_keys=input_keys, output_keys=output_keys, **kwargs)

  def call(self, *inputs, **unused_kwargs):
    """Run compute_output(), dense output layer, then split to a dictionary."""
    output = self.compute_output(*inputs)
    return split_to_dict(self.dense_out(output), self.output_splits)

  def compute_output(self, *inputs):
    """Takes tensors as input, runs network, and outputs a single tensor.

    Args:
      *inputs: A variable number of tensor inputs. Automatically infers
        self.input_keys from the name of each argmument in the function
        signature.

    Returns:
      A single tensor (usually [batch, time, channels]). The tensor can have any
      number of channels, because the base layer will run through a final dense
      layer to compress to appropriate number of channels for output_splits.
    """
    raise NotImplementedError


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


# ------------------ Straight-through Estimators -------------------------------
def straight_through_softmax(logits):
  """Straight-through estimator of a one-hot categorical distribution."""
  probs = tf.nn.softmax(logits)
  one_hot = tfp.distributions.OneHotCategorical(probs=probs)
  sample = tf.cast(one_hot.sample(), tf.float32)
  p_sample = probs * sample
  sample = tf.stop_gradient(sample - p_sample) + p_sample
  return sample, probs


def straight_through_choice(logits, values):
  """Straight-throgh estimator of choosing a value using a boolean mask."""
  choice, _ = straight_through_softmax(logits)
  return tf.reduce_sum(choice * values, axis=-1, keepdims=True)


def straight_through_int_quantization(x):
  """Rounds tensor to nearest integer using a straight through estimator.

  Args:
    x (tf.Tensor): Input Tensor that will get quantized. Values will be rounded
      to the nearest integer and are not assumed to be scaled (i.e., values in
      [-1.0, 1.0] will only produce -1, 0, or 1).

  Returns:
    A quantized version of the input Tensor `x`, with gradients as if no
    quantization happened.
  """
  return x + tf.stop_gradient(tf.math.round(x) - x)


# Masking ----------------------------------------------------------------------
def get_note_mask(q_pitch, max_regions=100, note_on_only=True):
  """Get a binary mask for each note from a monophonic instrument.

  Each transition of the value creates a new region. Returns the mask of each
  region.
  Args:
    q_pitch: A quantized value, such as pitch or velocity. Shape
      [batch, n_timesteps] or [batch, n_timesteps, 1].
    max_regions: Maximum number of note regions to consider in the sequence.
      Also, the channel dimension of the output mask. Each value transition
      defines a new region, e.g. each note-on and note-off count as a separate
      region.
    note_on_only: Return a mask that is true only for regions where the pitch
      is greater than 0.

  Returns:
    A binary mask of each region [batch, n_timesteps, max_regions].
  """
  # Only batch and time dimensions.
  if len(q_pitch.shape) == 3:
    q_pitch = q_pitch[:, :, 0]

  # Get onset and offset points.
  edges = tf.abs(spectral_ops.diff(q_pitch, axis=1)) > 0

  # Count endpoints as starts/ends of regions.
  edges = edges[:, :-1, ...]
  edges = tf.pad(edges,
                 [[0, 0], [1, 0]], mode='constant', constant_values=True)
  edges = tf.pad(edges,
                 [[0, 0], [0, 1]], mode='constant', constant_values=False)
  edges = tf.cast(edges, tf.int32)

  # Count up onset and offsets for each timestep.
  # Assumes each onset has a corresponding offset.
  # The -1 ensures that the 0th index is the first note.
  edge_idx = tf.cumsum(edges, axis=1) - 1

  # Create masks of shape [batch, n_timesteps, max_regions].
  note_mask = edge_idx[..., None] == tf.range(max_regions)[None, None, :]
  note_mask = tf.cast(note_mask, tf.float32)

  if note_on_only:
    # [batch, notes]
    note_pitches = get_note_moments(q_pitch, note_mask, return_std=False)
    # [batch, time, notes]
    note_on = tf.cast(note_pitches > 0.0, tf.float32)[:, None, :]
    # [batch, time, notes]
    note_mask *= note_on

  return note_mask


def get_note_lengths(note_mask):
  """Count the lengths of each note [batch, time, notes] -> [batch, notes]."""
  return tf.reduce_sum(note_mask, axis=1)


def get_note_moments(x, note_mask, return_std=True):
  """Return the moments of value xm, pooled over the length of the note.

  Args:
    x: Value to be pooled, [batch, time, dims] or [batch, time].
    note_mask: Binary mask of notes [batch, time, notes].
    return_std: Also return the standard deviation for each note.

  Returns:
    Values pooled over each note region, [batch, notes, dims] or [batch, notes].
    Returns only mean if return_std=False, else mean and std.
  """
  is_2d = len(x.shape) == 2
  if is_2d:
    x = x[:, :, tf.newaxis]

  note_mask_d = note_mask[..., tf.newaxis]  # [b, t, n, 1]
  note_lengths = tf.reduce_sum(note_mask_d, axis=1)  # [b, n, 1]

  # Mean.
  x_masked = x[:, :, tf.newaxis, :] * note_mask_d  # [b, t, n, d]
  x_mean = core.safe_divide(
      tf.reduce_sum(x_masked, axis=1), note_lengths)  # [b, n, d]

  # Standard Deviation.
  numerator = (x[:, :, tf.newaxis, :] -
               x_mean[:, tf.newaxis, :, :]) * note_mask_d
  numerator = tf.reduce_sum(numerator ** 2.0, axis=1)  # [b, n, d]
  x_std = core.safe_divide(numerator, note_lengths) ** 0.5

  x_mean = x_mean[:, :, 0] if is_2d else x_mean
  x_std = x_std[:, :, 0] if is_2d else x_std

  if return_std:
    return x_mean, x_std
  else:
    return x_mean


def pool_over_notes(x, note_mask):
  """Return the time-distributed average value of x pooled over the note.

  Args:
    x: Value to be pooled, [batch, time, dims].
    note_mask: Binary mask of notes [batch, time, notes].

  Returns:
    Values pooled over each note region, [batch, time, dims].
  """
  x_notes = get_note_moments(x, note_mask, return_std=False)  # [b, n, d]
  x_time_notes = (x_notes[:, tf.newaxis, ...] *
                  note_mask[..., tf.newaxis])  # [b, t, n, d]
  return tf.reduce_sum(x_time_notes, axis=2)  # [b, t, d]


def get_short_note_loss_mask(note_mask, note_lengths,
                             note_pitches, min_length=40):
  """Creates a 1-D binary mask for notes shorter than min_length."""
  short_notes = tf.logical_and(note_lengths < min_length, note_pitches > 0.0)
  short_notes = tf.cast(short_notes, tf.float32)
  short_note_mask = note_mask * short_notes[:, None, :]
  loss_mask = tf.reduce_sum(short_note_mask, axis=-1)
  return loss_mask


# ------------------ Normalization ---------------------------------------------
def normalize_op(x, norm_type='layer', eps=1e-5):
  """Apply either Group, Instance, or Layer normalization, or None."""
  if norm_type is not None:
    # mb, h, w, ch
    x_shape = tf.shape(x)

    n_groups = {'instance': x_shape[-1], 'layer': 1, 'group': 32}[norm_type]
    x = tf.reshape(
        x, tf.concat([x_shape[:-1], [n_groups, x_shape[-1] // n_groups]],
                     axis=0))

    mean, var = tf.nn.moments(x, [1, 2, 4], keepdims=True)
    x = (x - mean) / tf.sqrt(var + eps)
    x = tf.reshape(x, x_shape)
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


@gin.register
class RnnSandwich(tf.keras.Sequential):
  """RNN Sandwiched by two FC Stacks."""

  def __init__(self,
               fc_stack_ch=256,
               fc_stack_layers=2,
               rnn_ch=512,
               rnn_type='gru',
               **kwargs):
    layers = [
        FcStack(fc_stack_ch, fc_stack_layers),
        Rnn(rnn_ch, rnn_type),
        FcStack(fc_stack_ch, fc_stack_layers),
    ]
    super().__init__(layers, **kwargs)


