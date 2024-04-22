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

"""Library of effects functions."""

from ddsp import core
from ddsp import processors
from ddsp import synths
import gin
import tensorflow.compat.v2 as tf

tf_float32 = core.tf_float32


#------------------ Reverbs ----------------------------------------------------
@gin.register
class Reverb(processors.Processor):
  """Convolutional (FIR) reverb."""

  def __init__(self,
               trainable=False,
               reverb_length=48000,
               add_dry=True,
               name='reverb'):
    """Takes neural network outputs directly as the impulse response.

    Args:
      trainable: Learn the impulse_response as a single variable for the entire
        dataset.
      reverb_length: Length of the impulse response. Only used if
        trainable=True.
      add_dry: Add dry signal to reverberated signal on output.
      name: Name of processor module.
    """
    super().__init__(name=name, trainable=trainable)
    self._reverb_length = reverb_length
    self._add_dry = add_dry

  def _mask_dry_ir(self, ir):
    """Set first impulse response to zero to mask the dry signal."""
    # Make IR 2-D [batch, ir_size].
    if len(ir.shape) == 1:
      ir = ir[tf.newaxis, :]  # Add a batch dimension
    if len(ir.shape) == 3:
      ir = ir[:, :, 0]  # Remove unnessary channel dimension.
    # Mask the dry signal.
    dry_mask = tf.zeros([int(ir.shape[0]), 1], tf.float32)
    return tf.concat([dry_mask, ir[:, 1:]], axis=1)

  def _match_dimensions(self, audio, ir):
    """Tile the impulse response variable to match the batch size."""
    # Add batch dimension.
    if len(ir.shape) == 1:
      ir = ir[tf.newaxis, :]
    # Match batch dimension.
    batch_size = int(audio.shape[0])
    return tf.tile(ir, [batch_size, 1])

  def build(self, unused_input_shape):
    """Initialize impulse response."""
    if self.trainable:
      initializer = tf.random_normal_initializer(mean=0, stddev=1e-6)
      self._ir = self.add_weight(
          name='ir',
          shape=[self._reverb_length],
          dtype=tf.float32,
          initializer=initializer)
    self.built = True

  def get_controls(self, audio, ir=None):
    """Convert decoder outputs into ir response.

    Args:
      audio: Dry audio. 2-D Tensor of shape [batch, n_samples].
      ir: 3-D Tensor of shape [batch, ir_size, 1] or 2D Tensor of shape
        [batch, ir_size].

    Returns:
      controls: Dictionary of effect controls.

    Raises:
      ValueError: If trainable=False and ir is not provided.
    """
    if self.trainable:
      ir = self._match_dimensions(audio, self._ir)
    else:
      if ir is None:
        raise ValueError('Must provide "ir" tensor if Reverb trainable=False.')

    return {'audio': audio, 'ir': ir}

  def get_signal(self, audio, ir):
    """Apply impulse response.

    Args:
      audio: Dry audio, 2-D Tensor of shape [batch, n_samples].
      ir: 3-D Tensor of shape [batch, ir_size, 1] or 2D Tensor of shape
        [batch, ir_size].

    Returns:
      tensor of shape [batch, n_samples]
    """
    audio, ir = tf_float32(audio), tf_float32(ir)
    ir = self._mask_dry_ir(ir)
    wet = core.fft_convolve(audio, ir, padding='same', delay_compensation=0)
    return (wet + audio) if self._add_dry else wet


@gin.register
class ExpDecayReverb(Reverb):
  """Parameterize impulse response as a simple exponential decay."""

  def __init__(self,
               trainable=False,
               reverb_length=48000,
               scale_fn=core.exp_sigmoid,
               add_dry=True,
               name='exp_decay_reverb'):
    """Constructor.

    Args:
      trainable: Learn the impulse_response as a single variable for the entire
        dataset.
      reverb_length: Length of the impulse response.
      scale_fn: Function by which to scale the network outputs.
      add_dry: Add dry signal to reverberated signal on output.
      name: Name of processor module.
    """
    super().__init__(name=name, add_dry=add_dry, trainable=trainable)
    self._reverb_length = reverb_length
    self._scale_fn = scale_fn

  def _get_ir(self, gain, decay):
    """Simple exponential decay of white noise."""
    gain = self._scale_fn(gain)
    decay_exponent = 2.0 + tf.exp(decay)
    time = tf.linspace(0.0, 1.0, self._reverb_length)[tf.newaxis, :]
    noise = tf.random.uniform([1, self._reverb_length], minval=-1.0, maxval=1.0)
    ir = gain * tf.exp(-decay_exponent * time) * noise
    return ir

  def build(self, unused_input_shape):
    """Initialize impulse response."""
    if self.trainable:
      self._gain = self.add_weight(
          name='gain',
          shape=[1],
          dtype=tf.float32,
          initializer=tf.constant_initializer(2.0))
      self._decay = self.add_weight(
          name='decay',
          shape=[1],
          dtype=tf.float32,
          initializer=tf.constant_initializer(4.0))
    self.built = True

  def get_controls(self, audio, gain=None, decay=None):
    """Convert network outputs into ir response.

    Args:
      audio: Dry audio. 2-D Tensor of shape [batch, n_samples].
      gain: Linear gain of impulse response. Scaled by self._scale_fn.
        2D Tensor of shape [batch, 1]. Not used if trainable=True.
      decay: Exponential decay coefficient. The final impulse response is
        exp(-(2 + exp(decay)) * time) where time goes from 0 to 1.0 over the
        reverb_length samples. 2D Tensor of shape [batch, 1]. Not used if
        trainable=True.

    Returns:
      controls: Dictionary of effect controls.

    Raises:
      ValueError: If trainable=False and gain and decay are not provided.
    """
    if self.trainable:
      gain, decay = self._gain[tf.newaxis, :], self._decay[tf.newaxis, :]
    else:
      if gain is None or decay is None:
        raise ValueError('Must provide "gain" and "decay" tensors if '
                         'ExpDecayReverb trainable=False.')

    ir = self._get_ir(gain, decay)

    if self.trainable:
      ir = self._match_dimensions(audio, ir)

    return {'audio': audio, 'ir': ir}


@gin.register
class FilteredNoiseReverb(Reverb):
  """Parameterize impulse response with outputs of a filtered noise synth."""

  def __init__(self,
               trainable=False,
               reverb_length=48000,
               window_size=257,
               n_frames=1000,
               n_filter_banks=16,
               scale_fn=core.exp_sigmoid,
               initial_bias=-3.0,
               add_dry=True,
               name='filtered_noise_reverb'):
    """Constructor.

    Args:
      trainable: Learn the impulse_response as a single variable for the entire
        dataset.
      reverb_length: Length of the impulse response.
      window_size: Window size for filtered noise synthesizer.
      n_frames: Time resolution of magnitudes coefficients. Only used if
        trainable=True.
      n_filter_banks: Frequency resolution of magnitudes coefficients. Only used
        if trainable=True.
      scale_fn: Function by which to scale the magnitudes.
      initial_bias: Shift the filtered noise synth inputs by this amount
        (before scale_fn) to start generating noise in a resonable range when
        given magnitudes centered around 0.
      add_dry: Add dry signal to reverberated signal on output.
      name: Name of processor module.
    """
    super().__init__(name=name, add_dry=add_dry, trainable=trainable)
    self._n_frames = n_frames
    self._n_filter_banks = n_filter_banks
    self._synth = synths.FilteredNoise(n_samples=reverb_length,
                                       window_size=window_size,
                                       scale_fn=scale_fn,
                                       initial_bias=initial_bias)

  def build(self, unused_input_shape):
    """Initialize impulse response."""
    if self.trainable:
      initializer = tf.random_normal_initializer(mean=0, stddev=1e-2)
      self._magnitudes = self.add_weight(
          name='magnitudes',
          shape=[self._n_frames, self._n_filter_banks],
          dtype=tf.float32,
          initializer=initializer)
    self.built = True

  def get_controls(self, audio, magnitudes=None):
    """Convert network outputs into ir response.

    Args:
      audio: Dry audio. 2-D Tensor of shape [batch, n_samples].
      magnitudes: Magnitudes tensor of shape [batch, n_frames, n_filter_banks].
        Expects float32 that is strictly positive. Not used if trainable=True.

    Returns:
      controls: Dictionary of effect controls.

    Raises:
      ValueError: If trainable=False and magnitudes are not provided.
    """
    if self.trainable:
      magnitudes = self._magnitudes[tf.newaxis, :]
    else:
      if magnitudes is None:
        raise ValueError('Must provide "magnitudes" tensor if '
                         'FilteredNoiseReverb trainable=False.')

    ir = self._synth(magnitudes)

    if self.trainable:
      ir = self._match_dimensions(audio, ir)

    return {'audio': audio, 'ir': ir}


#------------------ Filters ----------------------------------------------------
@gin.register
class FIRFilter(processors.Processor):
  """Linear time-varying finite impulse response (LTV-FIR) filter."""

  def __init__(self,
               window_size=257,
               scale_fn=core.exp_sigmoid,
               name='fir_filter'):
    super().__init__(name=name)
    self.window_size = window_size
    self.scale_fn = scale_fn

  def get_controls(self, audio, magnitudes):
    """Convert network outputs into magnitudes response.

    Args:
      audio: Dry audio. 2-D Tensor of shape [batch, n_samples].
      magnitudes: 3-D Tensor of synthesizer parameters, of shape [batch, time,
        n_filter_banks].

    Returns:
      controls: Dictionary of tensors of synthesizer controls.
    """
    # Scale the magnitudes.
    if self.scale_fn is not None:
      magnitudes = self.scale_fn(magnitudes)

    return  {'audio': audio, 'magnitudes': magnitudes}

  def get_signal(self, audio, magnitudes):
    """Filter audio with LTV-FIR filter.

    Args:
      audio: Dry audio. 2-D Tensor of shape [batch, n_samples].
      magnitudes: Magnitudes tensor of shape [batch, n_frames, n_filter_banks].
        Expects float32 that is strictly positive.

    Returns:
      signal: Filtered audio of shape [batch, n_samples, 1].
    """
    return core.frequency_filter(audio,
                                 magnitudes,
                                 window_size=self.window_size)


#------------------ Modulation -------------------------------------------------
class ModDelay(processors.Processor):
  """Modulated delay times used in chorus, flanger, and vibrato effects."""

  def __init__(self,
               center_ms=15.0,
               depth_ms=10.0,
               sample_rate=16000,
               gain_scale_fn=core.exp_sigmoid,
               phase_scale_fn=tf.nn.sigmoid,
               add_dry=True,
               name='mod_delay'):
    super().__init__(name=name)
    self.center_ms = center_ms
    self.depth_ms = depth_ms
    self.sample_rate = sample_rate
    self.gain_scale_fn = gain_scale_fn
    self.phase_scale_fn = phase_scale_fn
    self.add_dry = add_dry

  def get_controls(self, audio, gain, phase):
    """Convert network outputs into magnitudes response.

    Args:
      audio: Dry audio. 2-D Tensor of shape [batch, n_samples].
      gain: Amplitude of modulated signal. Shape [batch_size, n_samples, 1].
      phase: Relative delay time. Shape [batch_size, n_samples, 1].

    Returns:
      controls: Dictionary of tensors of synthesizer controls.
    """
    if self.gain_scale_fn is not None:
      gain = self.gain_scale_fn(gain)

    if self.phase_scale_fn is not None:
      phase = self.phase_scale_fn(phase)

    return  {'audio': audio, 'gain': gain, 'phase': phase}

  def get_signal(self, audio, gain, phase):
    """Filter audio with LTV-FIR filter.

    Args:
      audio: Dry audio. 2-D Tensor of shape [batch, n_samples].
      gain: Amplitude of modulated signal. Shape [batch_size, n_samples, 1].
      phase: The normlaized instantaneous length of the delay, in the range of
        [center_ms - depth_ms, center_ms + depth_ms] from 0 to 1.0. Shape
        [batch_size, n_samples, 1].

    Returns:
      signal: Modulated audio of shape [batch, n_samples].
    """
    max_delay_ms = self.center_ms + self.depth_ms
    max_length_samples = int(self.sample_rate / 1000.0 * max_delay_ms)

    depth_phase = self.depth_ms / max_delay_ms
    center_phase = self.center_ms / max_delay_ms
    phase = phase * depth_phase + center_phase
    wet_audio = core.variable_length_delay(audio=audio,
                                           phase=phase,
                                           max_length=max_length_samples)
    # Remove channel dimension.
    if len(gain.shape) == 3:
      gain = gain[..., 0]

    wet_audio *= gain
    return (wet_audio + audio) if self.add_dry else wet_audio

