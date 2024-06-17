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

"""Library of neural network functions."""

import inspect

from ddsp import core
from ddsp import losses
import gin
import tensorflow as tf
import tensorflow_probability as tfp

tfk = tf.keras
tfkl = tfk.layers


# False positive lint error on tf.split().
# pylint: disable=redundant-keyword-arg


def gin_register_keras_layers():
  """Registers all keras layers and Sequential to be referenceable in gin."""
  # Register sequential model.
  gin.external_configurable(tf.keras.Sequential, 'tf.keras.Sequential')

  # Register all the layers.
  for k, v in inspect.getmembers(tf.keras.layers):
    # Duck typing for tf.keras.layers.Layer since keras uses metaclasses.
    if hasattr(v, 'variables'):
      gin.external_configurable(v, f'tf.keras.layers.{k}')


gin_register_keras_layers()


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
    if not input_keys:
      input_keys = self.get_argument_names('call')
      self.default_input_keys = list(self.get_default_argument_names('call'))
      self.default_input_values = list(self.get_default_argument_values('call'))
    else:
      # Manually specifying input keys overwrites default arguments.
      self.default_input_keys = []
      self.default_input_values = []
    output_keys = output_keys or self.get_return_annotations('call')

    self.input_keys = list(input_keys)
    self.output_keys = list(output_keys)

  @property
  def all_input_keys(self):
    """Full list of inputs and outputs."""
    return self.input_keys + self.default_input_keys

  @property
  def n_inputs(self):
    """Dynamically computed in case input_keys is changed in subclass init."""
    return len(self.all_input_keys)

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
    # Construct a list of input tensors equal in length and order to the `call`
    # input signature.
    # -- Start first with any tensor arguments.
    # -- Then lookup tensors from input dictionaries.
    # -- Use default values if not found.

    # Start by merging all dictionaries of tensors from the input.
    input_dict = {}
    for v in inputs:
      if isinstance(v, dict):
        input_dict.update(v)

    # And then strip all dictionaries from the input.
    inputs = [v for v in inputs if not isinstance(v, dict)]

    # Add any tensors from kwargs.
    for key in self.all_input_keys:
      if key in kwargs:
        input_dict[key] = kwargs[key]

    # And strip from kwargs.
    kwargs = {k: v for k, v in kwargs.items() if k not in self.all_input_keys}

    # Look up further inputs from the dictionaries.
    for key in self.input_keys:
      try:
        # If key is present use the input_dict value.
        inputs.append(core.nested_lookup(key, input_dict))
      except KeyError:
        # Skip if not present.
        pass

    # Add default arguments.
    for key, value in zip(self.default_input_keys, self.default_input_values):
      try:
        # If key is present, use the input_dict value.
        inputs.append(core.nested_lookup(key, input_dict))
      except KeyError:
        # Otherwise use the default value if not supplied as non-dict input.
        if len(inputs) < self.n_inputs:
          inputs.append(value)

    # Run input tensors through the model.
    if len(inputs) != self.n_inputs:
      raise TypeError(f'{len(inputs)} input tensors extracted from inputs'
                      '(including default args) but the layer expects '
                      f'{self.n_inputs} tensors.\n'
                      f'Input keys: {self.input_keys}\n'
                      f'Default keys: {self.default_input_keys}\n'
                      f'Default values: {self.default_input_values}\n'
                      f'Input dictionaries: {input_dict}\n'
                      f'Input Tensors (Args, Dicts, and Defaults): {inputs}\n')
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

  def get_default_argument_values(self, method):
    """Get list of strings for names of default arguments to method."""
    spec = inspect.getfullargspec(getattr(self, method))
    if spec.defaults:
      return spec.defaults
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


def get_nonlinearity(nonlinearity):
  """Get nonlinearity function by name."""
  try:
    return tf.keras.activations.get(nonlinearity)
  except ValueError:
    pass

  return getattr(tf.nn, nonlinearity)


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

  Each transition of the q_pitch value creates a new region. Returns the mask of
  each region.
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
  edges = tf.abs(core.diff(q_pitch, axis=1)) > 0

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


def get_note_mask_from_onset(q_pitch, onset, max_regions=100,
                             note_on_only=True):
  """Get a binary mask for each note from a monophonic instrument.

  Each onset creates a new region. Returns the mask of each region.
  Args:
    q_pitch: A quantized value, such as pitch or velocity. Shape
      [batch, n_timesteps] or [batch, n_timesteps, 1].
    onset: Binary onset in shape [batch, n_timesteps] or
    [batch, n_timesteps, 1]. 1 represents onset.
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
  if len(onset.shape) == 3:
    onset = onset[:, :, 0]

  edges = onset
  # Count endpoints as starts/ends of regions.
  edges = edges[:, 1:, ...]
  edges = tf.pad(edges,
                 [[0, 0], [1, 0]], mode='constant', constant_values=True)
  edges = tf.cast(edges, tf.int32)

  # Count up onset and offsets for each timestep.
  # Assumes each onset has a corresponding offset.
  # The -1 ensures that the 0th index is the first note.
  edge_idx = tf.cumsum(edges, axis=1) - 1

  # Create masks of shape [batch, n_timesteps, max_regions].
  note_mask = edge_idx[..., None] == tf.range(max_regions)[None, None, :]
  note_mask = tf.cast(note_mask, tf.float32)

  if note_on_only:
    # [batch, time, notes]
    note_on = tf.cast(q_pitch > 0.0, tf.float32)[:, :, None]
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


def pool_over_notes(x, note_mask, return_std=True):
  """Return the time-distributed average value of x pooled over the note.

  Args:
    x: Value to be pooled, [batch, time, dims].
    note_mask: Binary mask of notes [batch, time, notes].
    return_std: Also return the standard deviation for each note.

  Returns:
    Values pooled over each note region, [batch, time, dims].
    Returns only mean if return_std=False, else mean and std.
  """
  x_notes, x_notes_std = get_note_moments(x, note_mask,
                                          return_std=True)  # [b, n, d]
  x_time_notes_mean = (x_notes[:, tf.newaxis, ...] *
                       note_mask[..., tf.newaxis])  # [b, t, n, d]
  pooled_mean = tf.reduce_sum(x_time_notes_mean, axis=2)  # [b, t, d]

  if return_std:
    x_time_notes_std = (x_notes_std[:, tf.newaxis, ...] *
                        note_mask[..., tf.newaxis])  # [b, t, n, d]
    pooled_std = tf.reduce_sum(x_time_notes_std, axis=2)  # [b, t, d]
    return pooled_mean, pooled_std
  else:
    return pooled_mean


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


def get_norm(norm_type, conditional, shift_only):
  """Helper function to get conditional norm if needed."""
  if conditional:
    return ConditionalNorm(norm_type=norm_type, shift_only=shift_only)
  else:
    return Normalize(norm_type)


# ------------------ Resampling ------------------------------------------------
def polyphase_resample(x, stride=2, resample_type='down', trim_or_pad='pad'):
  """Resample by 'space_to_depth' conversion of time and channels.

  For example,
    Downsampling: [batch, time, ch] --> [batch, time/stride, ch*stride]
    Upsampling:   [batch, time, ch] --> [batch, time*stride, ch/stride]

  Named 'polyphase' resample because it performs a transformation similar to a
  polyphase filter (each "phase" gets its own channel, filter in parallel).
  Args:
    x: Input tensor, shape [batch, time, ch].
    stride: Amount to resample by.
    resample_type: 'up' or 'down'.
    trim_or_pad: 'trim' or 'pad'. What to do if time or channels cannot be
      evenly divided by stride.

  Returns:
    A resampled tensor.
  """
  is_4d = len(x.shape) == 4
  if is_4d:
    x = x[:, :, 0, :]

  n_time = x.shape[1]
  n_ch = x.shape[2]

  if resample_type == 'down':
    # Pad or trim.
    if trim_or_pad == 'pad':
      pad = (stride - n_time % stride) % stride
      x = tf.pad(x, [[0, 0], [0, pad], [0, 0]]) if pad > 0 else x
    else:
      trim = n_time % stride
      x = x[:, :-trim, :] if trim > 0 else x

    # Reshape.
    n_time = x.shape[1]
    x_reshape = tf.reshape(x, [-1, n_time // stride, n_ch * stride])

  elif resample_type == 'up':
    # Pad or trim.
    if trim_or_pad == 'pad':
      pad = (stride - n_ch % stride) % stride
      x = tf.pad(x, [[0, 0], [0, 0], [0, pad]]) if pad > 0 else x

    else:
      trim = n_ch % stride
      if trim > 0:
        x = x[:, :, :-trim]

    # Reshape.
    n_ch = x.shape[2]
    x_reshape = tf.reshape(x, [-1, n_time * stride, n_ch // stride])

  else:
    raise ValueError('`resample_type` must be either "up" or "down"')

  if is_4d:
    x_reshape = x_reshape[:, :, None, :]

  return x_reshape


@gin.register
class PolyphaseResample(tfkl.Layer):
  """Resample by interleaving time and channels."""

  def __init__(self,
               stride=2,
               resample_type='down',
               trim_or_pad='pad',
               **kwargs):
    super().__init__(**kwargs)
    self.stride = stride
    self.resample_type = resample_type
    self.trim_or_pad = trim_or_pad

  def call(self, x):
    return polyphase_resample(
        x, self.stride, self.resample_type, self.trim_or_pad)


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

  def __init__(self, ch, stride, shortcut, norm_type,
               conditional, shift_only, **kwargs):
    """Downsample frequency by stride, upsample channels by 4."""
    super().__init__(**kwargs)
    ch_out = 4 * ch
    self.shortcut = shortcut
    self.conditional = conditional

    # Layers.
    self.norm_input = get_norm(norm_type, conditional, shift_only)

    if self.shortcut:
      self.conv_proj = tfkl.Conv2D(
          ch_out, (1, 1), (1, stride), padding='same', name='conv_proj')
    layers = [
        tfkl.Conv2D(ch, (1, 1), (1, 1), padding='same'),
        NormReluConv(ch, 3, stride, norm_type),
        NormReluConv(ch_out, 1, 1, norm_type),
    ]
    self.bottleneck = tf.keras.Sequential(layers, name='bottleneck')

  def call(self, inputs):
    if self.conditional:
      x, z = inputs
      r = x

      x = ensure_4d(x)
      z = ensure_4d(z)
      x = tf.nn.relu(self.norm_input((x, z)))
    else:
      x = inputs
      r = x

      x = ensure_4d(x)
      x = tf.nn.relu(self.norm_input(x))

    # The projection shortcut should come after the first norm and ReLU
    # since it performs a 1x1 convolution.
    r = self.conv_proj(x) if self.shortcut else r
    x = self.bottleneck(x)
    return x + r


@gin.register
class ResidualStack(tfkl.Layer):
  """LayerNorm -> ReLU -> Conv layer."""

  def __init__(self,
               filters,
               block_sizes,
               strides,
               norm_type,
               conditional=False,
               shift_only=False,
               nonlinearity='relu',
               **kwargs):
    """ResNet layers."""
    super().__init__(**kwargs)
    self.conditional = conditional
    layers = []
    for (ch, n_layers, stride) in zip(filters, block_sizes, strides):

      # Only the first block per residual_stack uses shortcut and strides.
      layers.append(ResidualLayer(ch, stride, True, norm_type,
                                  conditional, shift_only))

      # Add the additional (n_layers - 1) layers to the stack.
      for _ in range(1, n_layers):
        layers.append(ResidualLayer(ch, 1, False, norm_type,
                                    conditional, shift_only))

    layers.append(Normalize(norm_type))
    layers.append(tfkl.Activation(get_nonlinearity(nonlinearity)))
    self.layers = layers

  def __call__(self, inputs):
    if self.conditional:
      x, z = inputs

    else:
      x = inputs

    for layer in self.layers:
      is_cond = self.conditional and isinstance(layer, ResidualLayer)
      l_in = [x, z] if is_cond else x
      x = layer(l_in)
    return x


@gin.register
class ResNet(tfkl.Layer):
  """Residual network."""

  def __init__(self, size='large', norm_type='layer',
               conditional=False, shift_only=False, **kwargs):
    super().__init__(**kwargs)
    self.conditional = conditional
    size_dict = {
        'small': (32, [2, 3, 4]),
        'medium': (32, [3, 4, 6]),
        'large': (64, [3, 4, 6]),
    }
    ch, blocks = size_dict[size]
    self.layers = [
        tfkl.Conv2D(64, (7, 7), (1, 2), padding='same'),
        tfkl.MaxPool2D(pool_size=(1, 3), strides=(1, 2), padding='same'),
        ResidualStack([ch, 2 * ch, 4 * ch], blocks, [1, 2, 2], norm_type,
                      conditional, shift_only),
        ResidualStack([8 * ch], [3], [2], norm_type,
                      conditional, shift_only)
    ]

  def __call__(self, inputs):
    if self.conditional:
      x, z = inputs

    else:
      x = inputs

    for layer in self.layers:
      is_cond = self.conditional and isinstance(layer, ResidualStack)
      l_in = [x, z] if is_cond else x
      x = layer(l_in)
    return x


# ---------------- Stacks ------------------------------------------------------
@gin.register
class Fc(tf.keras.Sequential):
  """Makes a Dense -> LayerNorm -> Leaky ReLU layer."""

  def __init__(self, ch=128, nonlinearity='leaky_relu', **kwargs):
    layers = [
        tfkl.Dense(ch),
        tfkl.LayerNormalization(),
        tfkl.Activation(get_nonlinearity(nonlinearity)),
    ]
    super().__init__(layers, **kwargs)


@gin.register
class FcStack(tf.keras.Sequential):
  """Stack Dense -> LayerNorm -> Leaky ReLU layers."""

  def __init__(self, ch=256, layers=2, nonlinearity='leaky_relu', **kwargs):
    layers = [Fc(ch, nonlinearity) for i in range(layers)]
    super().__init__(layers, **kwargs)


@gin.register
class Rnn(tfkl.Layer):
  """Single RNN layer."""

  def __init__(self, dims, rnn_type, return_sequences=True, bidir=False,
               **kwargs):
    super().__init__(**kwargs)
    rnn_class = {'lstm': tfkl.LSTM,
                 'gru': tfkl.GRU}[rnn_type]
    self.rnn = rnn_class(dims, return_sequences=return_sequences)
    if bidir:
      self.rnn = tfkl.Bidirectional(self.rnn)

  def call(self, x):
    return self.rnn(x)


@gin.register
class StatelessRnn(tfkl.Layer):
  """Stateless unidirectional RNN for streaming models."""

  def __init__(self, dims, rnn_type, **kwargs):
    super().__init__(**kwargs)
    rnn_class = {'lstm': tfkl.LSTM,
                 'gru': tfkl.GRU}[rnn_type]
    self.rnn = rnn_class(dims, return_sequences=True, return_state=True)

  def call(self, x, state):
    """Make a call with explicit carrying of state.

    Args:
      x: Input, shape [batch, T, dims_in].
      state: Last output, shape [batch, dims].

    Returns:
      y: Output, shape [batch, T, dims].
      new_state: Carried state, shape [batch, dims]
    """
    y, new_state = self.rnn(x, initial_state=state)
    return y, new_state


@gin.register
class RnnFc(tfk.Sequential):
  """RNN layer -> fully connected -> LayerNorm -> Activation fn."""

  def __init__(self, rnn_feat, out_feat,
               rnn_type='lstm', nonlinearity='sigmoid',
               bidir=False, n_rnn=1, **kwargs):
    layers = [Rnn(rnn_feat, rnn_type, bidir) for _ in range(n_rnn)]
    layers.append(Fc(out_feat, nonlinearity=nonlinearity))
    super().__init__(layers, **kwargs)


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


# ------------------ Utility Layers --------------------------------------------
@gin.register
class Identity(tfkl.Layer):
  """Utility identity layer."""

  def call(self, x):
    return x

gin.register(tfkl.Dense, module=__name__)


class SpectralNormalization(tf.keras.layers.Wrapper):
  """Performs spectral normalization on weights.

  Copied from soon to be deprecated TF addons that broke training.
  https://github.com/tensorflow/addons/issues/2807

  This wrapper controls the Lipschitz constant of the layer by
  constraining its spectral norm, which can stabilize the training of GANs.

  See [Spectral Normalization for Generative Adversarial
  Networks](https://arxiv.org/abs/1802.05957).

  Wrap `tf.keras.layers.Conv2D`:

  >>> x = np.random.rand(1, 10, 10, 1)
  >>> conv2d = SpectralNormalization(tf.keras.layers.Conv2D(2, 2))
  >>> y = conv2d(x)
  >>> y.shape
  TensorShape([1, 9, 9, 2])

  Wrap `tf.keras.layers.Dense`:

  >>> x = np.random.rand(1, 10, 10, 1)
  >>> dense = SpectralNormalization(tf.keras.layers.Dense(10))
  >>> y = dense(x)
  >>> y.shape
  TensorShape([1, 10, 10, 10])

    Args:
      layer: A `tf.keras.layers.Layer` instance that has either `kernel` or
        `embeddings` attribute.
      power_iterations: `int`, the number of iterations during normalization.

    Raises:
      AssertionError: If not initialized with a `Layer` instance.
      ValueError: If initialized with negative `power_iterations`.
      AttributeError: If `layer` does not has `kernel` or `embeddings`
      attribute.
  """

  def __init__(self,
               layer: tf.keras.layers,
               power_iterations: int = 1,
               **kwargs):
    super().__init__(layer, **kwargs)
    if power_iterations <= 0:
      raise ValueError('`power_iterations` should be greater than zero, got '
                       '`power_iterations={}`'.format(power_iterations))
    self.power_iterations = power_iterations
    self._initialized = False

  def build(self, input_shape):
    """Build `Layer`."""
    super().build(input_shape)
    input_shape = tf.TensorShape(input_shape)
    self.input_spec = tf.keras.layers.InputSpec(shape=[None] + input_shape[1:])

    if hasattr(self.layer, 'kernel'):
      self.w = self.layer.kernel
    elif hasattr(self.layer, 'embeddings'):
      self.w = self.layer.embeddings
    else:
      raise AttributeError('{} object has no attribute "kernel" nor '
                           '"embeddings"'.format(type(self.layer).__name__))

    self.w_shape = self.w.shape.as_list()

    self.u = self.add_weight(
        shape=(1, self.w_shape[-1]),
        initializer=tf.initializers.TruncatedNormal(stddev=0.02),
        trainable=False,
        name='sn_u',
        dtype=self.w.dtype,
    )

  def call(self, inputs, training=None):
    """Call `Layer`."""
    if training is None:
      training = tf.keras.backend.learning_phase()

    if training:
      self.normalize_weights()

    output = self.layer(inputs)
    return output

  def compute_output_shape(self, input_shape):
    return tf.TensorShape(
        self.layer.compute_output_shape(input_shape).as_list())

  @tf.function
  def normalize_weights(self):
    """Generate spectral normalized weights.

    This method will update the value of `self.w` with the
    spectral normalized value, so that the layer is ready for `call()`.
    """

    w = tf.reshape(self.w, [-1, self.w_shape[-1]])
    u = self.u

    with tf.name_scope('spectral_normalize'):
      for _ in range(self.power_iterations):
        v = tf.math.l2_normalize(tf.matmul(u, w, transpose_b=True))
        u = tf.math.l2_normalize(tf.matmul(v, w))

      sigma = tf.matmul(tf.matmul(v, w), u, transpose_b=True)

      self.w.assign(self.w / sigma)
      self.u.assign(u)

  def get_config(self):
    config = {'power_iterations': self.power_iterations}
    base_config = super().get_config()
    return {**base_config, **config}


# ------------------ Embeddings ------------------------------------------------
def get_embedding(vocab_size=1024, n_dims=256):
  """Get a real-valued embedding from an integer."""
  return tfkl.Embedding(
      input_dim=vocab_size,
      output_dim=n_dims,
      input_length=1)


# ------------------ Normalization ---------------------------------------------
class ConditionalScaleAndShift(tfkl.Layer):
  """Conditional scaling and shifting after normalization."""

  def __init__(self, shift_only=False, **kwargs):
    super().__init__(**kwargs)
    self.shift_only = shift_only

  def build(self, inputs_shapes):
    x_shape, _ = inputs_shapes
    self.x_ch = int(x_shape[-1])
    ch = self.x_ch if self.shift_only else 2 * self.x_ch
    self.dense = tfkl.Dense(ch)

  def call(self, inputs):
    """Conditional scaling and shifting after normalization."""
    x, z = inputs
    if self.shift_only:
      shift = self.dense(z)
      x += shift
    else:
      scale_shift = self.dense(z)
      scale = scale_shift[..., :self.x_ch]
      shift = scale_shift[..., self.x_ch:]
      x = (x * scale) + shift
    return x


@gin.register
class ConditionalNorm(tfkl.Layer):
  """Apply normalization and then conditional scale and shift."""

  def __init__(self,
               norm_type='instance',
               shift_only=False,
               **kwargs):
    """Apply normalization and then conditional scale and shift.

    Args:
      norm_type: Choose between 'group', 'instance', and 'layer' normalization.
      shift_only: Only apply a conditional shift, with no conditional scaling.
      **kwargs: Keras-specific constructor kwargs.
    """
    super().__init__(**kwargs)
    self.norm_type = norm_type
    self.conditional_scale_and_shift = ConditionalScaleAndShift(
        shift_only=shift_only)

  def call(self, inputs):
    """Apply normalization and then conditional scale and shift.

    Args:
      inputs: Pair of input tensors. X of shape [batch, height, width, ch], and
        z, conditioning tensor of a shape broadcastable to [batch, height,
        width, channels].

    Returns:
      Normalized and scaled output tensor of shape
          [batch, height, width, channels].
    """
    x, z = inputs
    x = normalize_op(x, norm_type=self.norm_type)
    return self.conditional_scale_and_shift([x, z])


# ------------------ Stacks ----------------------------------------------------
@gin.register
class SingleGru(tf.keras.Sequential):
  """Makes a GRU -> LayerNorm -> Dense network."""

  def __init__(self, gru_dim=128, **kwargs):
    layers = [
        tfkl.GRU(gru_dim, return_sequences=True),
        tfkl.LayerNormalization()
    ]
    super().__init__(layers, **kwargs)


@gin.register
class DilatedConvStack(tfkl.Layer):
  """Stack of dilated 1-D convolutions, optional conditioning at each layer."""

  def __init__(self,
               ch=256,
               layers_per_stack=5,
               stacks=2,
               kernel_size=3,
               dilation=2,
               norm_type=None,
               resample_type=None,
               resample_stride=1,
               stacks_per_resample=1,
               resample_after_convolve=True,
               spectral_norm=False,
               ortho_init=False,
               shift_only=False,
               conditional=False,
               **kwargs):
    """Constructor.

    Args:
      ch: Number of channels in each convolution layer.
      layers_per_stack: Convolution layers in each 'stack'. Dilation increases
        exponentially with layer depth inside a stack.
      stacks: Number of convolutions stacks.
      kernel_size: Size of convolution kernel.
      dilation: Exponent base of dilation factor within a stack.
      norm_type: Type of normalization before each nonlinearity, choose from
        'layer', 'instance', or 'group'.
      resample_type: Whether to 'upsample' or 'downsample' the signal. None
        performs no resampling.
      resample_stride: Stride for upsample or downsample layers.
      stacks_per_resample: Number of stacks per a resample layer.
      resample_after_convolve: Ordering of convolution and resampling. If True,
        apply `stacks_per_resample` stacks of convolution then a resampling
        layer. If False, apply the opposite order.
      spectral_norm: Apply spectral normalization to the convolution weights.
      ortho_init: Orthogonally initialize the kernel weights.
      shift_only: Learn/condition only shifts of normalization and not scale.
      conditional: Use conditioning signal to modulate shifts (and scales) of
        normalization (FiLM), instead of learned parameters.
      **kwargs: Other keras kwargs.

    Returns:
      Convolved and resampled signal. If inputs shape is [batch, time, ch_in],
      output shape is [batch, time_out, ch], where `ch` is the class kwarg, and
      `time_out` is (stacks // stacks_per_resample) * resample_stride times
      smaller or larger than `time` depending on whether `resample_type` is
      upsampling or downsampling.
    """
    super().__init__(**kwargs)
    self.conditional = conditional
    self.norm_type = norm_type
    self.resample_after_convolve = resample_after_convolve

    initializer = 'orthogonal' if ortho_init else 'glorot_uniform'

    def conv(ch, k, stride=1, dilation=1, transpose=False):
      """Make a convolution layer."""
      layer_class = tfkl.Conv2DTranspose if transpose else tfkl.Conv2D
      layer = layer_class(ch,
                          (k, 1),
                          (stride, 1),
                          dilation_rate=(dilation, 1),
                          padding='same',
                          kernel_initializer=initializer)
      if spectral_norm:
        return SpectralNormalization(layer)
      else:
        return layer

    # Layer Factories.
    def dilated_conv(i):
      """Generates a dilated convolution layer, based on `i` depth in stack."""
      if dilation > 0:
        dilation_rate = int(dilation ** i)
      else:
        # If dilation is negative, decrease dilation with depth instead of
        # increasing.
        dilation_rate = int((-dilation) ** (layers_per_stack - i - 1))
      layer = tf.keras.Sequential(name='dilated_conv')
      layer.add(tfkl.Activation(tf.nn.relu))
      layer.add(conv(ch, kernel_size, 1, dilation_rate))
      return layer

    def resample_layer():
      """Generates a resampling layer."""
      if resample_type == 'downsample':
        return conv(ch, resample_stride, resample_stride)
      elif resample_type == 'upsample':
        return conv(ch, resample_stride * 2, resample_stride, transpose=True)
      else:
        raise ValueError(f'invalid resample type: {resample_type}, '
                         'must be either `upsample` or `downsample`.')

    # Layers.
    self.conv_in = conv(ch, kernel_size)
    self.layers = []
    self.norms = []
    self.resample_layers = []

    # Stacks.
    for i in range(stacks):
      # Option: Resample before convolve.
      if (resample_type and not self.resample_after_convolve and
          i % stacks_per_resample == 0):
        self.resample_layers.append(resample_layer())

      # Convolve.
      for j in range(layers_per_stack):
        # Convolution.
        layer = dilated_conv(j)
        # Normalization / scale and shift.
        if self.conditional:
          norm = ConditionalNorm(norm_type=norm_type, shift_only=shift_only)
        else:
          norm = Normalize(norm_type=norm_type)

        # Add to the stack.
        self.layers.append(layer)
        self.norms.append(norm)

      # Option: Resample after convolve.
      if (resample_type and self.resample_after_convolve and
          (i + 1) % stacks_per_resample == 0):
        self.resample_layers.append(resample_layer())

    # For forward pass, calculate layers per a resample.
    if self.resample_layers:
      self.layers_per_resample = len(self.layers) // len(self.resample_layers)
    else:
      self.layers_per_resample = 0

  def call(self, inputs):
    """Forward pass."""
    # Get inputs.
    if self.conditional:
      x, z = inputs
      x = ensure_4d(x)
      z = ensure_4d(z)
    else:
      x = inputs
      x = ensure_4d(x)

    # Run them through the network.
    x = self.conv_in(x)

    # Stacks.
    for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):

      # Optional: Resample before conv.
      if (self.resample_layers and not self.resample_after_convolve and
          i  % self.layers_per_resample == 0):
        x = self.resample_layers[i // self.layers_per_resample](x)

      # Scale and shift by conditioning.
      if self.conditional:
        y = layer(x)
        x += norm([y, z])

      # Regular residual network.
      else:
        x += norm(layer(x))

      # Optional: Resample after conv.
      if (self.resample_layers and self.resample_after_convolve and
          (i + 1) % self.layers_per_resample == 0):
        x = self.resample_layers[i // self.layers_per_resample](x)

    return x[:, :, 0, :]  # Convert back to 3-D.


@gin.register
class FcStackOut(tfkl.Layer):
  """Stack of FC layers with variable hidden and output dims."""

  def __init__(self, ch, layers, n_out, **kwargs):
    super().__init__(**kwargs)
    self.stack = FcStack(ch, layers)
    self.dense_out = tfkl.Dense(n_out)

  def call(self, x):
    x = self.stack(x)
    return self.dense_out(x)


# ------------------ Vector Quantization ---------------------------------------
@gin.register
class VectorQuantization(tfkl.Layer):
  """Vector quantization using exponential moving average.

  Based on https://github.com/sarus-tech/tf2-published-models/blob/master/vqvae
  but with variables named to reflect https://arxiv.org/abs/1711.00937 and
  https://arxiv.org/abs/1906.00446
  """

  def __init__(self, k, gamma=0.99, restart_threshold=0.0, num_heads=1,
               commitment_loss_weight=0.2, **kwargs):
    super().__init__(**kwargs)
    self.k = k
    self.gamma = gamma
    self.restart_threshold = restart_threshold
    self.num_heads = num_heads
    self.commitment_loss_weight = commitment_loss_weight

  def build(self, input_shapes):
    self.depth = input_shapes[-1]
    if self.depth % self.num_heads != 0:
      raise ValueError('Input depth must be a multiple of the number of heads.')

    # Number of input vectors assigned to each cluster center.
    self.counts = tf.Variable(
        tf.zeros([self.k], dtype=tf.float32),
        trainable=False,
        name='counts',
        aggregation=tf.VariableAggregation.MEAN
    )

    # Sum of input vectors assigned to each cluster center.
    self.sums = tf.Variable(
        tf.zeros([self.k, self.depth // self.num_heads], dtype=tf.float32),
        trainable=False,
        name='sums',
        aggregation=tf.VariableAggregation.MEAN
    )

  def call(self, x, training=False):
    x_flat = tf.reshape(x, shape=(-1, self.depth))

    # Split each input vector into one segment per head.
    x_flat_split = tf.split(x_flat, self.num_heads, axis=1)
    x_flat = tf.concat(x_flat_split, axis=0)

    if training:
      # Figure out which centroids we want to keep, and which we want to
      # restart.
      n = x_flat.shape[0]
      keep = self.counts * self.k > self.restart_threshold * n
      restart = tf.math.logical_not(keep)

      # Replace centroids to restart with elements from the batch, using samples
      # from a uniform distribution as a fallback in case we need to restart
      # more centroids than we have elements in the batch.
      restart_idx = tf.squeeze(tf.where(restart), -1)
      n_replace = tf.minimum(tf.shape(restart_idx)[0], x_flat.shape[0])
      e_restart = tf.tensor_scatter_nd_update(
          tf.random.uniform([self.k, self.depth // self.num_heads]),
          tf.expand_dims(restart_idx[:n_replace], 1),
          tf.random.shuffle(x_flat)[:n_replace]
      )

      # Compute the values of the centroids we want to keep by dividing the
      # summed vectors by the corresponding counts.
      e = tf.where(
          tf.expand_dims(keep, 1),
          tf.math.divide_no_nan(self.sums, tf.expand_dims(self.counts, 1)),
          e_restart
      )

    else:
      # If not training, just use the centroids as is with no restarts.
      e = tf.math.divide_no_nan(self.sums, tf.expand_dims(self.counts, 1))

    # Compute distance between each input vector and each cluster center.
    distances = (
        tf.expand_dims(tf.reduce_sum(x_flat**2, axis=1), 1) -
        2 * tf.matmul(x_flat, tf.transpose(e)) +
        tf.expand_dims(tf.reduce_sum(e**2, axis=1), 0)
    )

    # Find nearest cluster center for each input vector.
    c = tf.argmin(distances, axis=1)

    # Quantize input vectors with straight-through estimator.
    z = tf.nn.embedding_lookup(e, c)
    z_split = tf.split(z, self.num_heads, axis=0)
    z = tf.concat(z_split, axis=1)
    z = tf.reshape(z, tf.shape(x))
    z = x + tf.stop_gradient(z - x)

    if training:
      # Compute cluster counts and vector sums over the batch.
      oh = tf.one_hot(indices=c, depth=self.k)
      counts = tf.reduce_sum(oh, axis=0)
      sums = tf.matmul(oh, x_flat, transpose_a=True)

      # Apply exponential moving average to cluster counts and vector sums.
      self.counts.assign_sub((1 - self.gamma) * (self.counts - counts))
      self.sums.assign_sub((1 - self.gamma) * (self.sums - sums))

    c_split = tf.split(c, self.num_heads, axis=0)
    c = tf.stack(c_split, axis=1)
    c = tf.reshape(c, tf.concat([tf.shape(x)[:-1], [self.num_heads]], axis=0))

    return z, c

  def unquantize(self, c):
    e = tf.math.divide_no_nan(self.sums, tf.expand_dims(self.counts, 1))
    z = tf.nn.embedding_lookup(e, c)
    return tf.reshape(z, tf.concat([tf.shape(c)[:-1], [self.depth]], axis=0))

  def committment_loss(self, z, z_q):
    """Encourage encoder to output embeddings close to the current centroids."""
    loss = losses.mean_difference(z, tf.stop_gradient(z_q), loss_type='L2')
    return self.commitment_loss_weight * loss

  def get_losses_dict(self, z, z_q):
    return {self.name + '_commitment_loss': self.committment_loss(z, z_q)}



