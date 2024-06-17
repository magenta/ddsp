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

"""Library of functions for differentiable digital signal processing (DDSP)."""

from collections import abc
import copy
from typing import Any, Dict, Optional, Sequence, Text, TypeVar

import gin
import numpy as np
from scipy import fftpack
import tensorflow.compat.v2 as tf

Number = TypeVar('Number', int, float, np.ndarray, tf.Tensor)
DB_RANGE = 80.0


# Utility Functions ------------------------------------------------------------
def tf_float32(x):
  """Ensure array/tensor is a float32 tf.Tensor."""
  if isinstance(x, tf.Tensor):
    return tf.cast(x, dtype=tf.float32)  # This is a no-op if x is float32.
  else:
    return tf.convert_to_tensor(x, tf.float32)


def make_iterable(x):
  """Wrap in a list if not iterable, return empty list if None."""
  if x is None:
    return []
  elif isinstance(x, (np.ndarray, tf.Tensor)):
    # Wrap in list so you don't iterate over the batch.
    return [x]
  else:
    return x if isinstance(x, abc.Iterable) else [x]


def to_dict(x, keys):
  """Converts list to a dictionary with supplied keys."""
  if isinstance(x, dict):
    # No-op for dict.
    return x
  else:
    # Wrap individual tensors in a list so we don't iterate over batch..
    x = make_iterable(x)
    if len(keys) != len(x):
      raise ValueError(f'Keys: {keys} must be the same length as {x}')
    # Use keys to create an output dictionary.
    return dict(zip(keys, x))


def copy_if_tf_function(x):
  """Copy if wrapped by tf.function.

  Prevents side-effects if x is the input to the tf.function and it is later
  altered. If eager, avoids unnecessary copies.
  Args:
    x: Any inputs.

  Returns:
    A shallow copy of x if inside a tf.function.
  """
  return copy.copy(x) if not tf.executing_eagerly() else x


def nested_keys(nested_dict: Dict[Text, Any],
                delimiter: Text = '/',
                prefix: Text = '') -> Sequence[Text]:
  """Returns a flattend list of nested key strings of a nested dict.

  Args:
    nested_dict: Nested dictionary.
    delimiter: String that splits the nested keys.
    prefix: Top-level key used for recursion, usually leave blank.

  Returns:
    List of nested key strings.
  """
  keys = []

  for k, v in nested_dict.items():
    key = k if not prefix else f'{prefix}{delimiter}{k}'

    if not isinstance(v, dict):
      keys.append(key)
    else:
      dict_keys = nested_keys(v, prefix=key)
      keys += dict_keys

  return keys


def nested_lookup(nested_key: Text,
                  nested_dict: Dict[Text, Any],
                  delimiter: Text = '/') -> tf.Tensor:
  """Returns the value of a nested dict according to a parsed input string.

  Args:
    nested_key: String of the form "key/key/key...".
    nested_dict: Nested dictionary.
    delimiter: String that splits the nested keys.

  Returns:
    value: Value of the key from the nested dictionary.
  """
  # Parse the input string.
  keys = nested_key.split(delimiter)
  # Return the nested value.
  value = nested_dict
  for key in keys:
    try:
      value = value[key]
    except KeyError:
      raise KeyError(f'Key \'{key}\' as a part of nested key \'{nested_key}\' '
                     'not found during nested dictionary lookup, out of '
                     f'available keys: {nested_keys(nested_dict)}')
  return value


def leaf_key(nested_key: Text,
             delimiter: Text = '/') -> tf.Tensor:
  """Returns the leaf node key name.

  Args:
    nested_key: String of the form "key/key/key...".
    delimiter: String that splits the nested keys.

  Returns:
    value: Final leaf node key name.
  """
  # Parse the input string.
  keys = nested_key.split(delimiter)
  return keys[-1]


def map_shape(x: Dict[Text, tf.Tensor]) -> Dict[Text, Sequence[int]]:
  """Recursively infer tensor shapes for a dictionary of tensors."""
  return tf.nest.map_structure(lambda t: list(tf.shape(t).numpy()), x)


def pad_axis(x, padding=(0, 0), axis=0, **pad_kwargs):
  """Pads only one axis of a tensor.

  Args:
    x: Input tensor.
    padding: Tuple of number of samples to pad (before, after).
    axis: Which axis to pad.
    **pad_kwargs: Other kwargs to pass to tf.pad.

  Returns:
    A tensor padded with padding along axis.
  """
  n_end_dims = len(x.shape) - axis - 1
  n_end_dims *= n_end_dims > 0
  paddings = [[0, 0]] * axis + [list(padding)] + [[0, 0]] * n_end_dims
  return tf.pad(x, paddings, **pad_kwargs)


def diff(x, axis=-1):
  """Take the finite difference of a tensor along an axis.

  Args:
    x: Input tensor of any dimension.
    axis: Axis on which to take the finite difference.

  Returns:
    d: Tensor with size less than x by 1 along the difference dimension.

  Raises:
    ValueError: Axis out of range for tensor.
  """
  shape = x.shape.as_list()
  ndim = len(shape)
  if axis >= ndim:
    raise ValueError('Invalid axis index: %d for tensor with only %d axes.' %
                     (axis, ndim))

  begin_back = [0 for _ in range(ndim)]
  begin_front = [0 for _ in range(ndim)]
  begin_front[axis] = 1

  shape[axis] -= 1
  slice_front = tf.slice(x, begin_front, shape)
  slice_back = tf.slice(x, begin_back, shape)
  d = slice_front - slice_back
  return d


# Math -------------------------------------------------------------------------
def nan_to_num(x, value=0.0):
  """Replace NaNs with value."""
  return tf.where(tf.math.is_nan(x), value * tf.ones_like(x), x)


def safe_divide(numerator, denominator, eps=1e-7):
  """Avoid dividing by zero by adding a small epsilon."""
  safe_denominator = tf.where(denominator == 0.0, eps, denominator)
  return numerator / safe_denominator


def safe_log(x, eps=1e-5):
  """Avoid taking the log of a non-positive number."""
  safe_x = tf.where(x <= 0.0, eps, x)
  return tf.math.log(safe_x)


def logb(x, base=2.0, eps=1e-5):
  """Logarithm with base as an argument."""
  return safe_divide(safe_log(x, eps), safe_log(base, eps), eps)


def log10(x, eps=1e-5):
  """Logarithm with base 10."""
  return logb(x, base=10, eps=eps)


def log_scale(x, min_x, max_x):
  """Scales a -1 to 1 value logarithmically between min and max."""
  x = tf_float32(x)
  x = (x + 1.0) / 2.0  # Scale [-1, 1] to [0, 1]
  return tf.exp((1.0 - x) * tf.math.log(min_x) + x * tf.math.log(max_x))


def soft_limit(x, x_min=0.0, x_max=1.0):
  """Softly limits inputs to the range [x_min, x_max]."""
  return tf.nn.softplus(x) + x_min - tf.nn.softplus(x - (x_max - x_min))


def gradient_reversal(x):
  """Identity operation that reverses the gradient."""
  return tf.stop_gradient(2.0 * x) - x


# Unit Conversions -------------------------------------------------------------
def amplitude_to_db(amplitude, ref_db=0.0, range_db=DB_RANGE, use_tf=True):
  """Converts amplitude in linear scale to power in decibels."""
  power = amplitude**2.0
  return power_to_db(power, ref_db=ref_db, range_db=range_db, use_tf=use_tf)


def power_to_db(power, ref_db=0.0, range_db=DB_RANGE, use_tf=True):
  """Converts power from linear scale to decibels."""
  # Choose library.
  maximum = tf.maximum if use_tf else np.maximum
  log_base10 = log10 if use_tf else np.log10

  # Convert to decibels.
  pmin = 10**-(range_db / 10.0)
  power = maximum(pmin, power)
  db = 10.0 * log_base10(power)

  # Set dynamic range.
  db -= ref_db
  db = maximum(db, -range_db)
  return db


def db_to_amplitude(db):
  """Converts power in decibels to amplitude in linear scale."""
  return db_to_power(db / 2.0)


def db_to_power(db):
  """Converts power from decibels to linear scale."""
  return 10.0**(db / 10.0)


def midi_to_hz(notes: Number, midi_zero_silence: bool = False) -> Number:
  """TF-compatible midi_to_hz function.

  Args:
    notes: Tensor containing encoded pitch in MIDI scale.
    midi_zero_silence: Whether to output 0 hz for midi 0, which would be
      convenient when midi 0 represents silence. By defualt (False), midi 0.0
      corresponds to 8.18 Hz.

  Returns:
    hz: Frequency of MIDI in hz, same shape as input.
  """
  notes = tf_float32(notes)
  hz = 440.0 * (2.0 ** ((notes - 69.0) / 12.0))
  # Map MIDI 0 as 0 hz when MIDI 0 is silence.
  if midi_zero_silence:
    hz = tf.where(tf.equal(notes, 0.0), 0.0, hz)
  return hz


def hz_to_midi(frequencies: Number) -> Number:
  """TF-compatible hz_to_midi function."""
  frequencies = tf_float32(frequencies)
  notes = 12.0 * (logb(frequencies, 2.0) - logb(440.0, 2.0)) + 69.0
  # Map 0 Hz to MIDI 0 (Replace -inf MIDI with 0.)
  notes = tf.where(tf.less_equal(frequencies, 0.0), 0.0, notes)
  return notes


def unit_to_midi(unit: Number,
                 midi_min: Number = 20.0,
                 midi_max: Number = 90.0,
                 clip: bool = False) -> Number:
  """Map the unit interval [0, 1] to MIDI notes."""
  unit = tf.clip_by_value(unit, 0.0, 1.0) if clip else unit
  return midi_min + (midi_max - midi_min) * unit


def midi_to_unit(midi: Number,
                 midi_min: Number = 20.0,
                 midi_max: Number = 90.0,
                 clip: bool = False) -> Number:
  """Map MIDI notes to the unit interval [0, 1]."""
  unit = (midi - midi_min) / (midi_max - midi_min)
  return tf.clip_by_value(unit, 0.0, 1.0) if clip else unit


def unit_to_hz(unit: Number,
               hz_min: Number,
               hz_max: Number,
               clip: bool = False) -> Number:
  """Map unit interval [0, 1] to [hz_min, hz_max], scaling logarithmically."""
  midi = unit_to_midi(unit,
                      midi_min=hz_to_midi(hz_min),
                      midi_max=hz_to_midi(hz_max),
                      clip=clip)
  return midi_to_hz(midi)


def hz_to_unit(hz: Number,
               hz_min: Number,
               hz_max: Number,
               clip: bool = False) -> Number:
  """Map [hz_min, hz_max] to unit interval [0, 1], scaling logarithmically."""
  midi = hz_to_midi(hz)
  return midi_to_unit(midi,
                      midi_min=hz_to_midi(hz_min),
                      midi_max=hz_to_midi(hz_max),
                      clip=clip)


def hz_to_bark(hz):
  """From Tranmuller (1990, https://asa.scitation.org/doi/10.1121/1.399849)."""
  return 26.81 / (1.0 + (1960.0 / hz)) - 0.53


def bark_to_hz(bark):
  """From Tranmuller (1990, https://asa.scitation.org/doi/10.1121/1.399849)."""
  return 1960.0 / (26.81 / (bark + 0.53) - 1.0)


def hz_to_mel(hz):
  """From Young et al. "The HTK book", Chapter 5.4."""
  return 2595.0 * logb(1.0 + hz / 700.0, 10.0)


def mel_to_hz(mel):
  """From Young et al. "The HTK book", Chapter 5.4."""
  return 700.0 * (10.0**(mel / 2595.0) - 1.0)


def hz_to_erb(hz):
  """Equivalent Rectangular Bandwidths (ERB) from Moore & Glasberg (1996).

  https://research.tue.nl/en/publications/a-revision-of-zwickers-loudness-model
  https://ccrma.stanford.edu/~jos/bbt/Equivalent_Rectangular_Bandwidth.html
  Args:
    hz: Inputs frequencies in hertz.

  Returns:
    Critical bandwidths (in hertz) for each input frequency.
  """
  return 0.108 * hz + 24.7


# Scaling functions ------------------------------------------------------------
@gin.register
def exp_sigmoid(x, exponent=10.0, max_value=2.0, threshold=1e-7):
  """Exponentiated Sigmoid pointwise nonlinearity.

  Bounds input to [threshold, max_value] with slope given by exponent.

  Args:
    x: Input tensor.
    exponent: In nonlinear regime (away from x=0), the output varies by this
      factor for every change of x by 1.0.
    max_value: Limiting value at x=inf.
    threshold: Limiting value at x=-inf. Stablizes training when outputs are
      pushed to 0.

  Returns:
    A tensor with pointwise nonlinearity applied.
  """
  x = tf_float32(x)
  return max_value * tf.nn.sigmoid(x)**tf.math.log(exponent) + threshold


@gin.register
def sym_exp_sigmoid(x, width=8.0):
  """Symmetrical version of exp_sigmoid centered at (0, 1e-7)."""
  x = tf_float32(x)
  return exp_sigmoid(width * (tf.abs(x)/2.0 - 1.0))


def _add_depth_axis(freqs: tf.Tensor, depth: int = 1) -> tf.Tensor:
  """Turns [batch, time, sinusoids*depth] to [batch, time, sinusoids, depth]."""
  freqs = freqs[..., tf.newaxis]
  # Unpack sinusoids dimension.
  n_batch, n_time, n_combined, _ = freqs.shape
  n_sinusoids = int(n_combined) // depth
  return tf.reshape(freqs, [n_batch, n_time, n_sinusoids, depth])


@gin.register
def frequencies_softmax(freqs: tf.Tensor,
                        depth: int = 1,
                        hz_min: float = 20.0,
                        hz_max: float = 8000.0) -> tf.Tensor:
  """Softmax to logarithmically scale network outputs to frequencies.

  Args:
    freqs: Neural network outputs, [batch, time, n_sinusoids * depth] or
      [batch, time, n_sinusoids, depth].
    depth: If freqs is 3-D, the number of softmax components per a sinusoid to
      unroll from the last dimension.
    hz_min: Lowest frequency to consider.
    hz_max: Highest frequency to consider.

  Returns:
    A tensor of frequencies in hertz [batch, time, n_sinusoids].
  """
  if len(freqs.shape) == 3:
    # Add depth: [B, T, N*D] -> [B, T, N, D]
    freqs = _add_depth_axis(freqs, depth)
  else:
    depth = int(freqs.shape[-1])

  # Probs: [B, T, N, D].
  f_probs = tf.nn.softmax(freqs, axis=-1)

  # [1, 1, 1, D]
  unit_bins = tf.linspace(0.0, 1.0, depth)
  unit_bins = unit_bins[tf.newaxis, tf.newaxis, tf.newaxis, :]

  # [B, T, N]
  f_unit = tf.reduce_sum(unit_bins * f_probs, axis=-1, keepdims=False)
  return unit_to_hz(f_unit, hz_min=hz_min, hz_max=hz_max)


@gin.register
def frequencies_sigmoid(freqs: tf.Tensor,
                        depth: int = 1,
                        hz_min: float = 0.0,
                        hz_max: float = 8000.0) -> tf.Tensor:
  """Sum of sigmoids to logarithmically scale network outputs to frequencies.

  Args:
    freqs: Neural network outputs, [batch, time, n_sinusoids * depth] or
      [batch, time, n_sinusoids, depth].
    depth: If freqs is 3-D, the number of sigmoid components per a sinusoid to
      unroll from the last dimension.
    hz_min: Lowest frequency to consider.
    hz_max: Highest frequency to consider.

  Returns:
    A tensor of frequencies in hertz [batch, time, n_sinusoids].
  """
  if len(freqs.shape) == 3:
    # Add depth: [B, T, N*D] -> [B, T, N, D]
    freqs = _add_depth_axis(freqs, depth)
  else:
    depth = int(freqs.shape[-1])

  # Probs: [B, T, N, D]
  f_probs = tf.nn.sigmoid(freqs)

  # [B, T N]
  # Partition frequency space in factors of 2, limit to range [hz_max, hz_min].
  hz_scales = []
  hz_min_copy = hz_min
  remainder = hz_max - hz_min
  scale_factor = remainder**(1.0 / depth)
  for i in range(depth):
    if i == (depth - 1):
      # Last depth element goes between minimum and remainder.
      hz_max = remainder
      hz_min = hz_min_copy
    else:
      # Reduce max by a constant factor for each depth element.
      hz_max = remainder * (1.0 - 1.0 / scale_factor)
      hz_min = 0
      remainder -= hz_max

    hz_scales.append(unit_to_hz(f_probs[..., i],
                                hz_min=hz_min,
                                hz_max=hz_max))

  return tf.reduce_sum(tf.stack(hz_scales, axis=-1), axis=-1)


@gin.register
def frequencies_critical_bands(freqs,
                               depth=1,
                               depth_scale=10.0,
                               bandwidth_scale=1.0,
                               hz_min=20.0,
                               hz_max=8000.0,
                               scale='bark'):
  """Center frequencies scaled on mel or bark scale, with ranges given by erb.

  Args:
    freqs: Neural network outputs, [batch, time, n_sinusoids * depth] or
      [batch, time, n_sinusoids, depth].
    depth: If freqs is 3-D, the number of sigmoid components per a sinusoid to
      unroll from the last dimension.
    depth_scale: The degree by which to reduce the influence of each subsequent
      dimension of depth.
    bandwidth_scale: Multiplier (to ERB) for the range of each sinusoid.
    hz_min: Lowest frequency to consider.
    hz_max: Highest frequency to consider.
    scale: Critical frequency scaling, must be either 'mel' or 'bark'.

  Returns:
    A tensor of frequencies in hertz [batch, time, n_sinusoids].
  """
  if len(freqs.shape) == 3:
    # Add depth: [B, T, N*D] -> [B, T, N, D]
    freqs = _add_depth_axis(freqs, depth)
  else:
    depth = int(freqs.shape[-1])

  # Figure out the number of sinusoids.
  n_sinusoids = freqs.shape[-2]

  # Initilaize the critical frequencies and bandwidths.
  if scale == 'bark':
    # Bark.
    bark_min = hz_to_bark(hz_min)
    bark_max = hz_to_bark(hz_max)
    linear_bark = np.linspace(bark_min, bark_max, n_sinusoids)
    f_center = bark_to_hz(linear_bark)
  else:
    # Mel.
    mel_min = hz_to_mel(hz_min)
    mel_max = hz_to_mel(hz_max)
    linear_mel = np.linspace(mel_min, mel_max, n_sinusoids)
    f_center = mel_to_hz(linear_mel)

  # Bandwiths given by equivalent rectangular bandwidth (ERB).
  bw = hz_to_erb(f_center)

  # Probs: [B, T, N, D]
  modifier = tf.nn.tanh(freqs)
  depth_modifier = depth_scale ** -tf.range(depth, dtype=tf.float32)
  # print(depth, depth_modifier, modifier)
  modifier = tf.reduce_sum(
      modifier * depth_modifier[tf.newaxis, tf.newaxis, tf.newaxis, :], axis=-1)

  f_modifier = bandwidth_scale * bw[tf.newaxis, tf.newaxis, :] * modifier
  return soft_limit(f_center + f_modifier, hz_min, hz_max)


# Resampling -------------------------------------------------------------------
def resample(inputs: tf.Tensor,
             n_timesteps: int,
             method: Text = 'linear',
             add_endpoint: bool = True) -> tf.Tensor:
  """Interpolates a tensor from n_frames to n_timesteps.

  Args:
    inputs: Framewise 1-D, 2-D, 3-D, or 4-D Tensor. Shape [n_frames],
      [batch_size, n_frames], [batch_size, n_frames, channels], or
      [batch_size, n_frames, n_freq, channels].
    n_timesteps: Time resolution of the output signal.
    method: Type of resampling, must be in ['nearest', 'linear', 'cubic',
      'window']. Linear and cubic ar typical bilinear, bicubic interpolation.
      'window' uses overlapping windows (only for upsampling) which is smoother
      for amplitude envelopes with large frame sizes.
    add_endpoint: Hold the last timestep for an additional step as the endpoint.
      Then, n_timesteps is divided evenly into n_frames segments. If false, use
      the last timestep as the endpoint, producing (n_frames - 1) segments with
      each having a length of n_timesteps / (n_frames - 1).
  Returns:
    Interpolated 1-D, 2-D, 3-D, or 4-D Tensor. Shape [n_timesteps],
      [batch_size, n_timesteps], [batch_size, n_timesteps, channels], or
      [batch_size, n_timesteps, n_freqs, channels].

  Raises:
    ValueError: If method is 'window' and input is 4-D.
    ValueError: If method is not one of 'nearest', 'linear', 'cubic', or
      'window'.
  """
  inputs = tf_float32(inputs)
  is_1d = len(inputs.shape) == 1
  is_2d = len(inputs.shape) == 2
  is_4d = len(inputs.shape) == 4

  # Ensure inputs are at least 3d.
  if is_1d:
    inputs = inputs[tf.newaxis, :, tf.newaxis]
  elif is_2d:
    inputs = inputs[:, :, tf.newaxis]

  def _image_resize(method):
    """Closure around tf.image.resize."""
    # Image resize needs 4-D input. Add/remove extra axis if not 4-D.
    outputs = inputs[:, :, tf.newaxis, :] if not is_4d else inputs
    outputs = tf.compat.v1.image.resize(outputs,
                                        [n_timesteps, outputs.shape[2]],
                                        method=method,
                                        align_corners=not add_endpoint)
    return outputs[:, :, 0, :] if not is_4d else outputs

  # Perform resampling.
  if method == 'nearest':
    outputs = _image_resize(tf.compat.v1.image.ResizeMethod.NEAREST_NEIGHBOR)
  elif method == 'linear':
    outputs = _image_resize(tf.compat.v1.image.ResizeMethod.BILINEAR)
  elif method == 'cubic':
    outputs = _image_resize(tf.compat.v1.image.ResizeMethod.BICUBIC)
  elif method == 'window':
    outputs = upsample_with_windows(inputs, n_timesteps, add_endpoint)
  else:
    raise ValueError('Method ({}) is invalid. Must be one of {}.'.format(
        method, "['nearest', 'linear', 'cubic', 'window']"))

  # Return outputs to the same dimensionality of the inputs.
  if is_1d:
    outputs = outputs[0, :, 0]
  elif is_2d:
    outputs = outputs[:, :, 0]

  return outputs


def upsample_with_windows(inputs: tf.Tensor,
                          n_timesteps: int,
                          add_endpoint: bool = True) -> tf.Tensor:
  """Upsample a series of frames using using overlapping hann windows.

  Good for amplitude envelopes.
  Args:
    inputs: Framewise 3-D tensor. Shape [batch_size, n_frames, n_channels].
    n_timesteps: The time resolution of the output signal.
    add_endpoint: Hold the last timestep for an additional step as the endpoint.
      Then, n_timesteps is divided evenly into n_frames segments. If false, use
      the last timestep as the endpoint, producing (n_frames - 1) segments with
      each having a length of n_timesteps / (n_frames - 1).

  Returns:
    Upsampled 3-D tensor. Shape [batch_size, n_timesteps, n_channels].

  Raises:
    ValueError: If input does not have 3 dimensions.
    ValueError: If attempting to use function for downsampling.
    ValueError: If n_timesteps is not divisible by n_frames (if add_endpoint is
      true) or n_frames - 1 (if add_endpoint is false).
  """
  inputs = tf_float32(inputs)

  if len(inputs.shape) != 3:
    raise ValueError('Upsample_with_windows() only supports 3 dimensions, '
                     'not {}.'.format(inputs.shape))

  # Mimic behavior of tf.image.resize.
  # For forward (not endpointed), hold value for last interval.
  if add_endpoint:
    inputs = tf.concat([inputs, inputs[:, -1:, :]], axis=1)

  n_frames = int(inputs.shape[1])
  n_intervals = (n_frames - 1)

  if n_frames >= n_timesteps:
    raise ValueError('Upsample with windows cannot be used for downsampling'
                     'More input frames ({}) than output timesteps ({})'.format(
                         n_frames, n_timesteps))

  if n_timesteps % n_intervals != 0.0:
    minus_one = '' if add_endpoint else ' - 1'
    raise ValueError(
        'For upsampling, the target the number of timesteps must be divisible '
        'by the number of input frames{}. (timesteps:{}, frames:{}, '
        'add_endpoint={}).'.format(minus_one, n_timesteps, n_frames,
                                   add_endpoint))

  # Constant overlap-add, half overlapping windows.
  hop_size = n_timesteps // n_intervals
  window_length = 2 * hop_size
  window = tf.signal.hann_window(window_length)  # [window]

  # Transpose for overlap_and_add.
  x = tf.transpose(inputs, perm=[0, 2, 1])  # [batch_size, n_channels, n_frames]

  # Broadcast multiply.
  # Add dimension for windows [batch_size, n_channels, n_frames, window].
  x = x[:, :, :, tf.newaxis]
  window = window[tf.newaxis, tf.newaxis, tf.newaxis, :]
  x_windowed = (x * window)
  x = tf.signal.overlap_and_add(x_windowed, hop_size)

  # Transpose back.
  x = tf.transpose(x, perm=[0, 2, 1])  # [batch_size, n_timesteps, n_channels]

  # Trim the rise and fall of the first and last window.
  return x[:, hop_size:-hop_size, :]


def center_crop(audio, frame_size):
  """Remove padding introduced from centering frames.

  Inverse of center_pad().
  Args:
    audio: Input, shape [batch, time, ...].
    frame_size: Size of each frame.

  Returns:
    audio_cropped: Shape [batch, time - (frame_size // 2) * 2, ...].
  """
  pad_amount = int(frame_size // 2)  # Symmetric even padding like librosa.
  return audio[:, pad_amount:-pad_amount]


# Synth conversions ------------------------------------------------------------
def sinusoidal_to_harmonic(sin_amps,
                           sin_freqs,
                           f0_hz,
                           harmonic_width=0.1,
                           n_harmonics=100,
                           sample_rate=16000,
                           normalize=False):
  """Extract harmonic components from sinusoids given a fundamental frequency.

  Args:
    sin_amps: Sinusoidal amplitudes (linear), shape [batch, time, n_sinusoids].
    sin_freqs: Sinusoidal frequencies in Hz, shape [batch, time, n_sinusoids].
    f0_hz: Fundamental frequency in Hz, shape [batch, time, 1].
    harmonic_width: Standard deviation of gaussian weighting based on relative
      frequency difference between a harmonic and a sinusoid.
    n_harmonics: Number of output harmonics to consider.
    sample_rate: Hertz, rate of the signal.
    normalize: If true, per timestep, each harmonic has a max of 1.0 weight to
      assign between the sinusoids. max(harm_amp) = max(sin_amp).

  Returns:
    harm_amp: Harmonic amplitude (linear), shape [batch, time, 1].
    harm_dist: Harmonic distribution, shape [batch, time, n_harmonics].
  """
  # [b, t, n_harm]
  harm_freqs = get_harmonic_frequencies(f0_hz, n_harmonics)

  # [b, t, n_harm, n_sin]
  freqs_diff = sin_freqs[:, :, tf.newaxis, :] - harm_freqs[..., tf.newaxis]
  freqs_ratio = tf.abs(safe_divide(freqs_diff, f0_hz[..., tf.newaxis]))
  weights = tf.math.exp(-(freqs_ratio / harmonic_width)**2.0)

  if normalize:
    # Sum of sinusoidal weights for a given harmonic. [b, t, n_harm, 1]
    weights_sum = tf.reduce_sum(weights, axis=-1, keepdims=True)
    weights_norm = safe_divide(weights, weights_sum)
    weights = tf.where(weights_sum > 1.0, weights_norm, weights)

  # [b, t, n_harm, n_sin] -> [b, t, n_harm]
  harm_amps = tf.reduce_sum(weights * sin_amps[:, :, tf.newaxis, :], axis=-1)

  # Filter harmonics above nyquist.
  harm_amps = remove_above_nyquist(harm_freqs, harm_amps, sample_rate)

  # Get harmonic distribution.
  harm_amp = tf.reduce_sum(harm_amps, axis=-1, keepdims=True)
  harm_dist = safe_divide(harm_amps, harm_amp)

  return harm_amp, harm_dist


def harmonic_to_sinusoidal(harm_amp, harm_dist, f0_hz, sample_rate=16000):
  """Converts controls for a harmonic synth to those for a sinusoidal synth."""
  n_harmonics = int(harm_dist.shape[-1])
  freqs = get_harmonic_frequencies(f0_hz, n_harmonics)
  # Double check to remove anything above Nyquist.
  harm_dist = remove_above_nyquist(freqs, harm_dist, sample_rate)
  # Renormalize after removing above nyquist.
  harm_dist_sum = tf.reduce_sum(harm_dist, axis=-1, keepdims=True)
  harm_dist = safe_divide(harm_dist, harm_dist_sum)
  amps = harm_amp * harm_dist
  return amps, freqs


# Harmonic Synthesizer ---------------------------------------------------------
# TODO(jesseengel): Remove reliance on global injection for angular cumsum.
@gin.configurable
def angular_cumsum(angular_frequency, chunk_size=1000):
  """Get phase by cumulative sumation of angular frequency.

  Custom cumsum splits first axis into chunks to avoid accumulation error.
  Just taking tf.sin(tf.cumsum(angular_frequency)) leads to accumulation of
  phase errors that are audible for long segments or at high sample rates. Also,
  in reduced precision settings, cumsum can overflow the threshold.

  During generation, if syntheiszed examples are longer than ~100k samples,
  consider using angular_sum to avoid noticible phase errors. This version is
  currently activated by global gin injection. Set the gin parameter
  `oscillator_bank.use_angular_cumsum=True` to activate.

  Given that we are going to take the sin of the accumulated phase anyways, we
  don't care about the phase modulo 2 pi. This code chops the incoming frequency
  into chunks, applies cumsum to each chunk, takes mod 2pi, and then stitches
  them back together by adding the cumulative values of the final step of each
  chunk to the next chunk.

  Seems to be ~30% faster on CPU, but at least 40% slower on TPU.

  Args:
    angular_frequency: Radians per a sample. Shape [batch, time, ...].
      If there is no batch dimension, one will be temporarily added.
    chunk_size: Number of samples per a chunk. to avoid overflow at low
       precision [chunk_size <= (accumulation_threshold / pi)].

  Returns:
    The accumulated phase in range [0, 2*pi], shape [batch, time, ...].
  """
  # Get tensor shapes.
  n_batch = angular_frequency.shape[0]
  n_time = angular_frequency.shape[1]
  n_dims = len(angular_frequency.shape)
  n_ch_dims = n_dims - 2

  # Pad if needed.
  remainder = n_time % chunk_size
  if remainder:
    pad_amount = chunk_size - remainder
    angular_frequency = pad_axis(angular_frequency, [0, pad_amount], axis=1)

  # Split input into chunks.
  length = angular_frequency.shape[1]
  n_chunks = int(length / chunk_size)
  chunks = tf.reshape(angular_frequency,
                      [n_batch, n_chunks, chunk_size] + [-1] * n_ch_dims)
  phase = tf.cumsum(chunks, axis=2)

  # Add offsets.
  # Offset of the next row is the last entry of the previous row.
  offsets = phase[:, :, -1:, ...] % (2.0 * np.pi)
  offsets = pad_axis(offsets, [1, 0], axis=1)
  offsets = offsets[:, :-1, ...]

  # Offset is cumulative among the rows.
  offsets = tf.cumsum(offsets, axis=1) % (2.0 * np.pi)
  phase = phase + offsets

  # Put back in original shape.
  phase = phase % (2.0 * np.pi)
  phase = tf.reshape(phase, [n_batch, length] + [-1] * n_ch_dims)

  # Remove padding if added it.
  if remainder:
    phase = phase[:, :n_time]
  return phase


def remove_above_nyquist(frequency_envelopes: tf.Tensor,
                         amplitude_envelopes: tf.Tensor,
                         sample_rate: int = 16000) -> tf.Tensor:
  """Set amplitudes for oscillators above nyquist to 0.

  Args:
    frequency_envelopes: Sample-wise oscillator frequencies (Hz). Shape
      [batch_size, n_samples, n_sinusoids].
    amplitude_envelopes: Sample-wise oscillator amplitude. Shape [batch_size,
      n_samples, n_sinusoids].
    sample_rate: Sample rate in samples per a second.

  Returns:
    amplitude_envelopes: Sample-wise filtered oscillator amplitude.
      Shape [batch_size, n_samples, n_sinusoids].
  """
  frequency_envelopes = tf_float32(frequency_envelopes)
  amplitude_envelopes = tf_float32(amplitude_envelopes)

  amplitude_envelopes = tf.where(
      tf.greater_equal(frequency_envelopes, sample_rate / 2.0),
      tf.zeros_like(amplitude_envelopes), amplitude_envelopes)
  return amplitude_envelopes


def normalize_harmonics(harmonic_distribution, f0_hz=None, sample_rate=None):
  """Normalize the harmonic distribution, optionally removing above nyquist."""
  # Bandlimit the harmonic distribution.
  if sample_rate is not None and f0_hz is not None:
    n_harmonics = int(harmonic_distribution.shape[-1])
    harmonic_frequencies = get_harmonic_frequencies(f0_hz, n_harmonics)
    harmonic_distribution = remove_above_nyquist(
        harmonic_frequencies, harmonic_distribution, sample_rate)

  # Normalize
  harmonic_distribution = safe_divide(
      harmonic_distribution,
      tf.reduce_sum(harmonic_distribution, axis=-1, keepdims=True))
  return harmonic_distribution


# TODO(jesseengel): Remove reliance on global injection for angular cumsum.
@gin.configurable
def oscillator_bank(frequency_envelopes: tf.Tensor,
                    amplitude_envelopes: tf.Tensor,
                    sample_rate: int = 16000,
                    sum_sinusoids: bool = True,
                    use_angular_cumsum: bool = False) -> tf.Tensor:
  """Generates audio from sample-wise frequencies for a bank of oscillators.

  Args:
    frequency_envelopes: Sample-wise oscillator frequencies (Hz). Shape
      [batch_size, n_samples, n_sinusoids].
    amplitude_envelopes: Sample-wise oscillator amplitude. Shape [batch_size,
      n_samples, n_sinusoids].
    sample_rate: Sample rate in samples per a second.
    sum_sinusoids: Add up audio from all the sinusoids.
    use_angular_cumsum: If synthesized examples are longer than ~100k audio
      samples, consider use_angular_cumsum to avoid accumulating noticible phase
      errors due to the limited precision of tf.cumsum. Unlike the rest of the
      library, this property can be set with global dependency injection with
      gin. Set the gin parameter `oscillator_bank.use_angular_cumsum=True`
      to activate. Avoids accumulation of errors for generation, but don't use
      usually for training because it is slower on accelerators.

  Returns:
    wav: Sample-wise audio. Shape [batch_size, n_samples, n_sinusoids] if
      sum_sinusoids=False, else shape is [batch_size, n_samples].
  """
  frequency_envelopes = tf_float32(frequency_envelopes)
  amplitude_envelopes = tf_float32(amplitude_envelopes)

  # Don't exceed Nyquist.
  amplitude_envelopes = remove_above_nyquist(frequency_envelopes,
                                             amplitude_envelopes,
                                             sample_rate)

  # Angular frequency, Hz -> radians per sample.
  omegas = frequency_envelopes * (2.0 * np.pi)  # rad / sec
  omegas = omegas / float(sample_rate)  # rad / sample

  # Accumulate phase and synthesize.
  if use_angular_cumsum:
    # Avoids accumulation errors.
    phases = angular_cumsum(omegas)
  else:
    phases = tf.cumsum(omegas, axis=1)

  # Convert to waveforms.
  wavs = tf.sin(phases)
  audio = amplitude_envelopes * wavs  # [mb, n_samples, n_sinusoids]
  if sum_sinusoids:
    audio = tf.reduce_sum(audio, axis=-1)  # [mb, n_samples]
  return audio


# TODO(jesseengel): Remove reliance on global injection for angular cumsum.
@gin.configurable
def harmonic_oscillator_bank(
    frequency: tf.Tensor,
    amplitude_envelopes: tf.Tensor,
    initial_phase: Optional[tf.Tensor] = None,
    sample_rate: int = 16000,
    use_angular_cumsum: bool = True) -> tf.Tensor:
  """Special oscillator bank for harmonic frequencies and streaming synthesis.


  Args:
    frequency: Sample-wise oscillator frequencies (Hz). Shape
      [batch_size, n_samples, 1].
    amplitude_envelopes: Sample-wise oscillator amplitude. Shape [batch_size,
      n_samples, n_sinusoids].
    initial_phase: Starting phase. Shape [batch_size, 1, 1].
    sample_rate: Sample rate in samples per a second.
    use_angular_cumsum: If synthesized examples are longer than ~100k audio
      samples, consider use_angular_cumsum to avoid accumulating noticible phase
      errors due to the limited precision of tf.cumsum. Unlike the rest of the
      library, this property can be set with global dependency injection with
      gin. Set the gin parameter `oscillator_bank.use_angular_cumsum=True`
      to activate. Avoids accumulation of errors for generation, but don't use
      usually for training because it is slower on accelerators.

  Returns:
    wav: Sample-wise audio. Shape [batch_size, n_samples, n_sinusoids] if
      sum_sinusoids=False, else shape is [batch_size, n_samples].
  """
  frequency = tf_float32(frequency)
  amplitude_envelopes = tf_float32(amplitude_envelopes)

  # Angular frequency, Hz -> radians per sample.
  omega = frequency * (2.0 * np.pi)  # rad / sec
  omega = omega / float(sample_rate)  # rad / sample

  # Accumulate phase and synthesize.
  if use_angular_cumsum:
    # Avoids accumulation errors.
    phases = angular_cumsum(omega)
  else:
    phases = tf.cumsum(omega, axis=1)

  if initial_phase is None:
    initial_phase = tf.zeros([phases.shape[0], 1, 1])

  phases += initial_phase
  final_phase = phases[:, -1:, 0:1]

  n_harmonics = int(amplitude_envelopes.shape[-1])
  f_ratios = tf.linspace(1.0, float(n_harmonics), int(n_harmonics))
  f_ratios = f_ratios[tf.newaxis, tf.newaxis, :]
  phases = phases * f_ratios

  # Convert to waveforms.
  wavs = tf.sin(phases)
  audio = amplitude_envelopes * wavs  # [mb, n_samples, n_sinusoids]
  audio = tf.reduce_sum(audio, axis=-1)  # [mb, n_samples]

  return audio, final_phase


def get_harmonic_frequencies(frequencies: tf.Tensor,
                             n_harmonics: int) -> tf.Tensor:
  """Create integer multiples of the fundamental frequency.

  Args:
    frequencies: Fundamental frequencies (Hz). Shape [batch_size, :, 1].
    n_harmonics: Number of harmonics.

  Returns:
    harmonic_frequencies: Oscillator frequencies (Hz).
      Shape [batch_size, :, n_harmonics].
  """
  frequencies = tf_float32(frequencies)

  f_ratios = tf.linspace(1.0, float(n_harmonics), int(n_harmonics))
  f_ratios = f_ratios[tf.newaxis, tf.newaxis, :]
  harmonic_frequencies = frequencies * f_ratios
  return harmonic_frequencies


def harmonic_synthesis(frequencies: tf.Tensor,
                       amplitudes: tf.Tensor,
                       harmonic_shifts: Optional[tf.Tensor] = None,
                       harmonic_distribution: Optional[tf.Tensor] = None,
                       n_samples: int = 64000,
                       sample_rate: int = 16000,
                       amp_resample_method: Text = 'window',
                       use_angular_cumsum: bool = False) -> tf.Tensor:
  """Generate audio from frame-wise monophonic harmonic oscillator bank.

  Args:
    frequencies: Frame-wise fundamental frequency in Hz. Shape [batch_size,
      n_frames, 1].
    amplitudes: Frame-wise oscillator peak amplitude. Shape [batch_size,
      n_frames, 1].
    harmonic_shifts: Harmonic frequency variations (Hz), zero-centered. Total
      frequency of a harmonic is equal to (frequencies * harmonic_number * (1 +
      harmonic_shifts)). Shape [batch_size, n_frames, n_harmonics].
    harmonic_distribution: Harmonic amplitude variations, ranged zero to one.
      Total amplitude of a harmonic is equal to (amplitudes *
      harmonic_distribution). Shape [batch_size, n_frames, n_harmonics].
    n_samples: Total length of output audio. Interpolates and crops to this.
    sample_rate: Sample rate.
    amp_resample_method: Mode with which to resample amplitude envelopes.
    use_angular_cumsum: Use angular cumulative sum on accumulating phase
      instead of tf.cumsum. More accurate for inference.

  Returns:
    audio: Output audio. Shape [batch_size, n_samples, 1]
  """
  frequencies = tf_float32(frequencies)
  amplitudes = tf_float32(amplitudes)

  if harmonic_distribution is not None:
    harmonic_distribution = tf_float32(harmonic_distribution)
    n_harmonics = int(harmonic_distribution.shape[-1])
  elif harmonic_shifts is not None:
    harmonic_shifts = tf_float32(harmonic_shifts)
    n_harmonics = int(harmonic_shifts.shape[-1])
  else:
    n_harmonics = 1

  # Create harmonic frequencies [batch_size, n_frames, n_harmonics].
  harmonic_frequencies = get_harmonic_frequencies(frequencies, n_harmonics)
  if harmonic_shifts is not None:
    harmonic_frequencies *= (1.0 + harmonic_shifts)

  # Create harmonic amplitudes [batch_size, n_frames, n_harmonics].
  if harmonic_distribution is not None:
    harmonic_amplitudes = amplitudes * harmonic_distribution
  else:
    harmonic_amplitudes = amplitudes

  # Create sample-wise envelopes.
  frequency_envelopes = resample(harmonic_frequencies, n_samples)  # cycles/sec
  amplitude_envelopes = resample(harmonic_amplitudes, n_samples,
                                 method=amp_resample_method)

  # Synthesize from harmonics [batch_size, n_samples].
  audio = oscillator_bank(frequency_envelopes,
                          amplitude_envelopes,
                          sample_rate=sample_rate,
                          use_angular_cumsum=use_angular_cumsum)
  return audio


def streaming_harmonic_synthesis(
    frequencies: tf.Tensor,
    amplitudes: tf.Tensor,
    harmonic_distribution: Optional[tf.Tensor] = None,
    initial_phase: Optional[tf.Tensor] = None,
    n_samples: int = 64000,
    sample_rate: int = 16000,
    amp_resample_method: Text = 'linear') -> tf.Tensor:
  """Generate audio from frame-wise monophonic harmonic oscillator bank.

  Args:
    frequencies: Frame-wise fundamental frequency in Hz. Shape [batch_size,
      n_frames, 1].
    amplitudes: Frame-wise oscillator peak amplitude. Shape [batch_size,
      n_frames, 1].
    harmonic_distribution: Harmonic amplitude variations, ranged zero to one.
      Total amplitude of a harmonic is equal to (amplitudes *
      harmonic_distribution). Shape [batch_size, n_frames, n_harmonics].
    initial_phase: Starting phase. Shape [batch_size, 1, 1].
    n_samples: Total length of output audio. Interpolates and crops to this.
    sample_rate: Sample rate.
    amp_resample_method: Mode with which to resample amplitude envelopes.

  Returns:
    audio: Output audio. Shape [batch_size, n_samples, 1]
  """
  frequencies = tf_float32(frequencies)
  amplitudes = tf_float32(amplitudes)

  if harmonic_distribution is not None:
    # Create harmonic amplitudes [batch_size, n_frames, n_harmonics].
    harmonic_distribution = tf_float32(harmonic_distribution)
    # Don't exceed Nyquist.
    harmonic_distribution = normalize_harmonics(
        harmonic_distribution, frequencies, sample_rate)
    harmonic_amplitudes = amplitudes * harmonic_distribution
  else:
    harmonic_amplitudes = amplitudes

  # Create sample-wise envelopes.
  frequencies = resample(frequencies, n_samples)  # cycles/sec
  amplitude_envelopes = resample(harmonic_amplitudes, n_samples,
                                 method=amp_resample_method)

  # Synthesize from harmonics [batch_size, n_samples].
  audio, final_phase = harmonic_oscillator_bank(
      frequencies,
      amplitude_envelopes,
      initial_phase,
      sample_rate=sample_rate)
  return audio, final_phase


# Wavetable Synthesizer --------------------------------------------------------
def linear_lookup(phase: tf.Tensor,
                  wavetables: tf.Tensor) -> tf.Tensor:
  """Lookup from wavetables with linear interpolation.

  Args:
    phase: The instantaneous phase of the base oscillator, ranging from 0 to
      1.0. This gives the position to lookup in the wavetable.
      Shape [batch_size, n_samples, 1].
    wavetables: Wavetables to be read from on lookup. Shape [batch_size,
      n_samples, n_wavetable] or [batch_size, n_wavetable].

  Returns:
    The resulting audio from linearly interpolated lookup of the wavetables at
      each point in time. Shape [batch_size, n_samples].
  """
  phase, wavetables = tf_float32(phase), tf_float32(wavetables)

  # Add a time dimension if not present.
  if len(wavetables.shape) == 2:
    wavetables = wavetables[:, tf.newaxis, :]

  # Add a wavetable dimension if not present.
  if len(phase.shape) == 2:
    phase = phase[:, :, tf.newaxis]

  # Add first sample to end of wavetable for smooth linear interpolation
  # between the last point in the wavetable and the first point.
  wavetables = tf.concat([wavetables, wavetables[..., 0:1]], axis=-1)
  n_wavetable = int(wavetables.shape[-1])

  # Get a phase value for each point on the wavetable.
  phase_wavetables = tf.linspace(0.0, 1.0, n_wavetable)

  # Get pair-wise distances from the oscillator phase to each wavetable point.
  # Axes are [batch, time, n_wavetable].
  phase_distance = tf.abs((phase - phase_wavetables[tf.newaxis, tf.newaxis, :]))

  # Put distance in units of wavetable samples.
  phase_distance *= n_wavetable - 1

  # Weighting for interpolation.
  # Distance is > 1.0 (and thus weights are 0.0) for all but nearest neighbors.
  weights = tf.nn.relu(1.0 - phase_distance)
  weighted_wavetables = weights * wavetables

  # Interpolated audio from summing the weighted wavetable at each timestep.
  return tf.reduce_sum(weighted_wavetables, axis=-1)


def harmonic_distribution_to_wavetable(harmonic_distribution, n_wavetable=2048):
  """Convert a harmonic distribution into a wavetable for synthesis.

  Args:
    harmonic_distribution: Shape [batch, time, n_harmonics], where the last axis
      is normalized (sums to 1.0).
    n_wavetable: Number of samples to have in the wavetable. If more than the
      number of harmonics, performs interpolation/upsampling.

  Returns:
    A series of wavetables, shape [batch, time, n_wavetable]
  """
  n_harmonics = harmonic_distribution.shape[-1]
  n_pad = int(n_wavetable/2 - n_harmonics)
  # Pad the left for DC component, pad the right for wavetable interpolation.
  fft_in = tf.pad(harmonic_distribution, [[0, 0], [0, 0], [1, n_pad]])
  fft_in = tf.complex(fft_in, tf.zeros_like(fft_in))
  wavetable = tf.signal.irfft(fft_in) * (n_wavetable / 2)
  return wavetable


def wavetable_synthesis(frequencies: tf.Tensor,
                        amplitudes: tf.Tensor,
                        wavetables: tf.Tensor,
                        n_samples: int = 64000,
                        sample_rate: int = 16000):
  """Monophonic wavetable synthesizer.

  Args:
    frequencies: Frame-wise frequency in Hertz of the fundamental oscillator.
      Shape [batch_size, n_frames, 1].
    amplitudes: Frame-wise amplitude envelope to apply to the oscillator. Shape
      [batch_size, n_frames, 1].
    wavetables: Frame-wise wavetables from which to lookup. Shape
      [batch_size, n_wavetable] or [batch_size, n_frames, n_wavetable].
    n_samples: Total length of output audio. Interpolates and crops to this.
    sample_rate: Number of samples per a second.

  Returns:
    audio: Audio at the frequency and amplitude of the inputs, with harmonics
      given by the wavetable. Shape [batch_size, n_samples].
  """
  wavetables = tf_float32(wavetables)

  # Create sample-wise envelopes.
  amplitude_envelope = resample(amplitudes, n_samples, method='window')[:, :, 0]
  frequency_envelope = resample(frequencies, n_samples)  # cycles / sec

  # Create intermediate wavetables.
  wavetable_shape = wavetables.shape.as_list()
  if len(wavetable_shape) == 3 and wavetable_shape[1] > 1:
    wavetables = resample(wavetables, n_samples)

  # Accumulate phase (in cycles which range from 0.0 to 1.0).
  phase_velocity = frequency_envelope / float(sample_rate)  # cycles / sample

  # Note: Cumsum accumulates _very_ small errors at float32 precision.
  # On the order of milli-Hertz.
  phase = tf.cumsum(phase_velocity, axis=1, exclusive=True) % 1.0

  # Synthesize with linear lookup.
  audio = linear_lookup(phase, wavetables)

  # Modulate with amplitude envelope.
  audio *= amplitude_envelope
  return audio


def variable_length_delay(phase: tf.Tensor,
                          audio: tf.Tensor,
                          max_length: int = 512) -> tf.Tensor:
  """Delay audio by a time-vaying amount using linear interpolation.

  Useful for modulation effects such as vibrato, chorus, and flanging.
  Args:
    phase: The normlaized instantaneous length of the delay, ranging from 0 to
      1.0. This corresponds to a delay of 0 to max_length samples. Shape
      [batch_size, n_samples, 1].
    audio: Audio signal to be delayed. Shape [batch_size, n_samples].
    max_length: Maximimum delay in samples.

  Returns:
    The delayed audio signal. Shape [batch_size, n_samples].
  """
  phase, audio = tf_float32(phase), tf_float32(audio)

  # Make causal by zero-padding audio up front.
  audio = tf.pad(audio, [(0, 0), (max_length - 1, 0)])
  # Cut audio up into frames of max_length.
  frames = tf.signal.frame(audio,
                           frame_length=max_length,
                           frame_step=1,
                           pad_end=False)
  # Reverse frames so that [0, 1] phase corresponds to [0, max_length] delay.
  frames = frames[..., ::-1]
  # Read audio from the past frames.
  return linear_lookup(phase, frames)


# Time-varying convolution -----------------------------------------------------
def get_fft_size(frame_size: int, ir_size: int, power_of_2: bool = True) -> int:
  """Calculate final size for efficient FFT.

  Args:
    frame_size: Size of the audio frame.
    ir_size: Size of the convolving impulse response.
    power_of_2: Constrain to be a power of 2. If False, allow other 5-smooth
      numbers. TPU requires power of 2, while GPU is more flexible.

  Returns:
    fft_size: Size for efficient FFT.
  """
  convolved_frame_size = ir_size + frame_size - 1
  if power_of_2:
    # Next power of 2.
    fft_size = int(2**np.ceil(np.log2(convolved_frame_size)))
  else:
    fft_size = int(fftpack.helper.next_fast_len(convolved_frame_size))
  return fft_size


def crop_and_compensate_delay(audio: tf.Tensor, audio_size: int, ir_size: int,
                              padding: Text,
                              delay_compensation: int) -> tf.Tensor:
  """Crop audio output from convolution to compensate for group delay.

  Args:
    audio: Audio after convolution. Tensor of shape [batch, time_steps].
    audio_size: Initial size of the audio before convolution.
    ir_size: Size of the convolving impulse response.
    padding: Either 'valid' or 'same'. For 'same' the final output to be the
      same size as the input audio (audio_timesteps). For 'valid' the audio is
      extended to include the tail of the impulse response (audio_timesteps +
      ir_timesteps - 1).
    delay_compensation: Samples to crop from start of output audio to compensate
      for group delay of the impulse response. If delay_compensation < 0 it
      defaults to automatically calculating a constant group delay of the
      windowed linear phase filter from frequency_impulse_response().

  Returns:
    Tensor of cropped and shifted audio.

  Raises:
    ValueError: If padding is not either 'valid' or 'same'.
  """
  # Crop the output.
  if padding == 'valid':
    crop_size = ir_size + audio_size - 1
  elif padding == 'same':
    crop_size = audio_size
  else:
    raise ValueError('Padding must be \'valid\' or \'same\', instead '
                     'of {}.'.format(padding))

  # Compensate for the group delay of the filter by trimming the front.
  # For an impulse response produced by frequency_impulse_response(),
  # the group delay is constant because the filter is linear phase.
  total_size = int(audio.shape[-1])
  crop = total_size - crop_size
  start = ((ir_size - 1) // 2 -
           1 if delay_compensation < 0 else delay_compensation)
  end = crop - start
  return audio[:, start:-end]


def fft_convolve(audio: tf.Tensor,
                 impulse_response: tf.Tensor,
                 padding: Text = 'same',
                 delay_compensation: int = -1) -> tf.Tensor:
  """Filter audio with frames of time-varying impulse responses.

  Time-varying filter. Given audio [batch, n_samples], and a series of impulse
  responses [batch, n_frames, n_impulse_response], splits the audio into frames,
  applies filters, and then overlap-and-adds audio back together.
  Applies non-windowed non-overlapping STFT/ISTFT to efficiently compute
  convolution for large impulse response sizes.

  Args:
    audio: Input audio. Tensor of shape [batch, audio_timesteps].
    impulse_response: Finite impulse response to convolve. Can either be a 2-D
      Tensor of shape [batch, ir_size], or a 3-D Tensor of shape [batch,
      ir_frames, ir_size]. A 2-D tensor will apply a single linear
      time-invariant filter to the audio. A 3-D Tensor will apply a linear
      time-varying filter. Automatically chops the audio into equally shaped
      blocks to match ir_frames.
    padding: Either 'valid' or 'same'. For 'same' the final output to be the
      same size as the input audio (audio_timesteps). For 'valid' the audio is
      extended to include the tail of the impulse response (audio_timesteps +
      ir_timesteps - 1).
    delay_compensation: Samples to crop from start of output audio to compensate
      for group delay of the impulse response. If delay_compensation is less
      than 0 it defaults to automatically calculating a constant group delay of
      the windowed linear phase filter from frequency_impulse_response().

  Returns:
    audio_out: Convolved audio. Tensor of shape
        [batch, audio_timesteps + ir_timesteps - 1] ('valid' padding) or shape
        [batch, audio_timesteps] ('same' padding).

  Raises:
    ValueError: If audio and impulse response have different batch size.
    ValueError: If audio cannot be split into evenly spaced frames. (i.e. the
      number of impulse response frames is on the order of the audio size and
      not a multiple of the audio size.)
  """
  audio, impulse_response = tf_float32(audio), tf_float32(impulse_response)

  # Get shapes of audio.
  batch_size, audio_size = audio.shape.as_list()

  # Add a frame dimension to impulse response if it doesn't have one.
  ir_shape = impulse_response.shape.as_list()
  if len(ir_shape) == 2:
    impulse_response = impulse_response[:, tf.newaxis, :]

  # Broadcast impulse response.
  if ir_shape[0] == 1 and batch_size > 1:
    impulse_response = tf.tile(impulse_response, [batch_size, 1, 1])

  # Get shapes of impulse response.
  ir_shape = impulse_response.shape.as_list()
  batch_size_ir, n_ir_frames, ir_size = ir_shape

  # Validate that batch sizes match.
  if batch_size != batch_size_ir:
    raise ValueError('Batch size of audio ({}) and impulse response ({}) must '
                     'be the same.'.format(batch_size, batch_size_ir))

  # Cut audio into frames.
  frame_size = int(np.ceil(audio_size / n_ir_frames))
  hop_size = frame_size
  audio_frames = tf.signal.frame(audio, frame_size, hop_size, pad_end=True)

  # Check that number of frames match.
  n_audio_frames = int(audio_frames.shape[1])
  if n_audio_frames != n_ir_frames:
    raise ValueError(
        'Number of Audio frames ({}) and impulse response frames ({}) do not '
        'match. For small hop size = ceil(audio_size / n_ir_frames), '
        'number of impulse response frames must be a multiple of the audio '
        'size.'.format(n_audio_frames, n_ir_frames))

  # Pad and FFT the audio and impulse responses.
  fft_size = get_fft_size(frame_size, ir_size, power_of_2=True)
  audio_fft = tf.signal.rfft(audio_frames, [fft_size])
  ir_fft = tf.signal.rfft(impulse_response, [fft_size])

  # Multiply the FFTs (same as convolution in time).
  audio_ir_fft = tf.multiply(audio_fft, ir_fft)

  # Take the IFFT to resynthesize audio.
  audio_frames_out = tf.signal.irfft(audio_ir_fft)
  audio_out = tf.signal.overlap_and_add(audio_frames_out, hop_size)

  # Crop and shift the output audio.
  return crop_and_compensate_delay(audio_out, audio_size, ir_size, padding,
                                   delay_compensation)


# Filter Design ----------------------------------------------------------------
def apply_window_to_impulse_response(impulse_response: tf.Tensor,
                                     window_size: int = 0,
                                     causal: bool = False) -> tf.Tensor:
  """Apply a window to an impulse response and put in causal form.

  Args:
    impulse_response: A series of impulse responses frames to window, of shape
      [batch, n_frames, ir_size].
    window_size: Size of the window to apply in the time domain. If window_size
      is less than 1, it defaults to the impulse_response size.
    causal: Impulse responnse input is in causal form (peak in the middle).

  Returns:
    impulse_response: Windowed impulse response in causal form, with last
      dimension cropped to window_size if window_size is greater than 0 and less
      than ir_size.
  """
  impulse_response = tf_float32(impulse_response)

  # If IR is in causal form, put it in zero-phase form.
  if causal:
    impulse_response = tf.signal.fftshift(impulse_response, axes=-1)

  # Get a window for better time/frequency resolution than rectangular.
  # Window defaults to IR size, cannot be bigger.
  ir_size = int(impulse_response.shape[-1])
  if (window_size <= 0) or (window_size > ir_size):
    window_size = ir_size
  window = tf.signal.hann_window(window_size)

  # Zero pad the window and put in in zero-phase form.
  padding = ir_size - window_size
  if padding > 0:
    half_idx = (window_size + 1) // 2
    window = tf.concat([window[half_idx:],
                        tf.zeros([padding]),
                        window[:half_idx]], axis=0)
  else:
    window = tf.signal.fftshift(window, axes=-1)

  # Apply the window, to get new IR (both in zero-phase form).
  window = tf.broadcast_to(window, impulse_response.shape)
  impulse_response = window * tf.math.real(impulse_response)

  # Put IR in causal form and trim zero padding.
  if padding > 0:
    first_half_start = (ir_size - (half_idx - 1)) + 1
    second_half_end = half_idx + 1
    impulse_response = tf.concat([impulse_response[..., first_half_start:],
                                  impulse_response[..., :second_half_end]],
                                 axis=-1)
  else:
    impulse_response = tf.signal.fftshift(impulse_response, axes=-1)

  return impulse_response


def frequency_impulse_response(magnitudes: tf.Tensor,
                               window_size: int = 0) -> tf.Tensor:
  """Get windowed impulse responses using the frequency sampling method.

  Follows the approach in:
  https://ccrma.stanford.edu/~jos/sasp/Windowing_Desired_Impulse_Response.html

  Args:
    magnitudes: Frequency transfer curve. Float32 Tensor of shape [batch,
      n_frames, n_frequencies] or [batch, n_frequencies]. The frequencies of the
      last dimension are ordered as [0, f_nyqist / (n_frequencies -1), ...,
      f_nyquist], where f_nyquist is (sample_rate / 2). Automatically splits the
      audio into equally sized frames to match frames in magnitudes.
    window_size: Size of the window to apply in the time domain. If window_size
      is less than 1, it defaults to the impulse_response size.

  Returns:
    impulse_response: Time-domain FIR filter of shape
      [batch, frames, window_size] or [batch, window_size].

  Raises:
    ValueError: If window size is larger than fft size.
  """
  # Get the IR (zero-phase form).
  magnitudes = tf.complex(magnitudes, tf.zeros_like(magnitudes))
  impulse_response = tf.signal.irfft(magnitudes)

  # Window and put in causal form.
  impulse_response = apply_window_to_impulse_response(impulse_response,
                                                      window_size)

  return impulse_response


def sinc(x, threshold=1e-20):
  """Normalized zero phase version (peak at zero)."""
  x = tf_float32(x)
  x = tf.where(tf.abs(x) < threshold, threshold * tf.ones_like(x), x)
  x = np.pi * x
  return tf.sin(x) / x


def sinc_impulse_response(cutoff_frequency,
                          window_size=512,
                          sample_rate=None,
                          high_pass=False):
  """Get a sinc impulse response for a set of low-pass cutoff frequencies.

  Args:
    cutoff_frequency: Frequency cutoff for low-pass sinc filter. If the
      sample_rate is given, cutoff_frequency is in Hertz. If sample_rate is
      None, cutoff_frequency is normalized ratio (frequency/nyquist) in the
      range [0, 1.0]. Shape [batch_size, n_time, 1].
    window_size: Size of the Hamming window to apply to the impulse.
    sample_rate: Optionally provide the sample rate.
    high_pass: If true, filter removes frequencies below cutoff (high-pass), if
      false [default], filter removes frequencies above cutoff (low-pass).

  Returns:
    impulse_response: A series of impulse responses. Shape
      [batch_size, n_time, (window_size // 2) * 2 + 1].
  """
  # Convert frequency to samples/sample_rate [0, Nyquist] -> [0, 1].
  if sample_rate is not None:
    cutoff_frequency *= 2.0 / float(sample_rate)

  # Create impulse response axis.
  half_size = window_size // 2
  full_size = half_size * 2 + 1
  idx = tf.range(-half_size, half_size + 1.0, dtype=tf.float32)
  idx = idx[tf.newaxis, tf.newaxis, :]

  # Compute impulse response.
  impulse_response = sinc(cutoff_frequency * idx)

  # Window the impulse response.
  window = tf.signal.hamming_window(full_size)
  window = tf.broadcast_to(window, impulse_response.shape)
  impulse_response = window * tf.math.real(impulse_response)

  # Normalize for unity gain.
  impulse_response /= tf.abs(
      tf.reduce_sum(impulse_response, axis=-1, keepdims=True))

  if high_pass:
    # Invert filter.
    pass_through = np.zeros(impulse_response.shape)
    pass_through[..., half_size] = 1.0
    pass_through = tf.convert_to_tensor(pass_through, dtype=tf.float32)
    impulse_response = pass_through - impulse_response

  return impulse_response


def frequency_filter(audio: tf.Tensor,
                     magnitudes: tf.Tensor,
                     window_size: int = 0,
                     padding: Text = 'same') -> tf.Tensor:
  """Filter audio with a finite impulse response filter.

  Args:
    audio: Input audio. Tensor of shape [batch, audio_timesteps].
    magnitudes: Frequency transfer curve. Float32 Tensor of shape [batch,
      n_frames, n_frequencies] or [batch, n_frequencies]. The frequencies of the
      last dimension are ordered as [0, f_nyqist / (n_frequencies -1), ...,
      f_nyquist], where f_nyquist is (sample_rate / 2). Automatically splits the
      audio into equally sized frames to match frames in magnitudes.
    window_size: Size of the window to apply in the time domain. If window_size
      is less than 1, it is set as the default (n_frequencies).
    padding: Either 'valid' or 'same'. For 'same' the final output to be the
      same size as the input audio (audio_timesteps). For 'valid' the audio is
      extended to include the tail of the impulse response (audio_timesteps +
      window_size - 1).

  Returns:
    Filtered audio. Tensor of shape
        [batch, audio_timesteps + window_size - 1] ('valid' padding) or shape
        [batch, audio_timesteps] ('same' padding).
  """
  impulse_response = frequency_impulse_response(magnitudes,
                                                window_size=window_size)
  return fft_convolve(audio, impulse_response, padding=padding)


def sinc_filter(audio: tf.Tensor,
                cutoff_frequency: tf.Tensor,
                window_size: int = 512,
                sample_rate: Optional[int] = None,
                padding: Text = 'same',
                high_pass: bool = False) -> tf.Tensor:
  """Filter audio with sinc low-pass filter.

  Args:
    audio: Input audio. Tensor of shape [batch, audio_timesteps].
    cutoff_frequency: Frequency cutoff for low-pass sinc filter. If the
      sample_rate is given, cutoff_frequency is in Hertz. If sample_rate is
      None, cutoff_frequency is normalized ratio (frequency/nyquist) in the
      range [0, 1.0]. Shape [batch_size, n_time, 1].
    window_size: Size of the Hamming window to apply to the impulse.
    sample_rate: Optionally provide the sample rate.
    padding: Either 'valid' or 'same'. For 'same' the final output to be the
      same size as the input audio (audio_timesteps). For 'valid' the audio is
      extended to include the tail of the impulse response (audio_timesteps +
      window_size - 1).
    high_pass: If true, filter removes frequencies below cutoff (high-pass), if
      false [default], filter removes frequencies above cutoff (low-pass).

  Returns:
    Filtered audio. Tensor of shape
      [batch, audio_timesteps + window_size - 1] ('valid' padding) or shape
      [batch, audio_timesteps] ('same' padding).
  """
  impulse_response = sinc_impulse_response(cutoff_frequency,
                                           window_size=window_size,
                                           sample_rate=sample_rate,
                                           high_pass=high_pass)
  return fft_convolve(audio, impulse_response, padding=padding)
