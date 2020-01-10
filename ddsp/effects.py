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
"""Library of effects functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ddsp import core
from ddsp import processors
import gin
import tensorflow.compat.v1 as tf

tf_float32 = core.tf_float32


#------------------ Reverberation ----------------------------------------------
@gin.register
class Reverb(processors.Processor):
  """Convolutional (FIR) reverb."""

  def __init__(self, reverb_only=False, name='reverb'):
    """Takes neural network outputs directly as the impulse response.

    Args:
      reverb_only: Only output the reverberated signal. If False, the output is
        a linear combination of the dry and wet signals.
      name: Name of processor module.
    """
    super(Reverb, self).__init__(name=name)
    self._reverb_only = reverb_only

  def _mask_dry_ir(self, ir):
    """Set first impulse response to zero to mask the dry signal."""
    # Make IR 2-D [batch, ir_size].
    if len(ir.shape) == 1:
      ir = ir[tf.newaxis, :]  # Add a batch dimension
    if len(ir.shape) == 3:
      ir = ir[:, :, 0]  # Remove unnessary channel dimension.
    # Mask the dry signal.
    dry_ir = tf.zeros([int(ir.shape[0]), 1], tf.float32)
    return tf.concat([dry_ir, ir[:, 1:]], axis=1)

  def get_controls(self, audio, nn_outputs):
    """Convert decoder outputs into ir response.

    Args:
      audio: Dry audio. 2-D Tensor of shape [batch, n_samples].
      nn_outputs: 3-D Tensor of shape [batch, ir_size, 1] or 2D Tensor of shape
        [batch, ir_size].

    Returns:
      controls: Dictionary of effect controls.
    """
    return {'audio': audio, 'ir': nn_outputs}

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
    return wet if self._reverb_only else (wet + audio)


@gin.register
class TrainableReverb(Reverb):
  """Learn a single impulse response for the whole dataset."""

  def __init__(self,
               reverb_length=64000,
               reverb_only=False,
               name='trainable_reverb'):
    """Constructor.

    Args:
      reverb_length: Length of the impulse response.
      reverb_only: Only output the reverberated signal.
      name: Name of processor module.
    """
    super(TrainableReverb, self).__init__(name=name, reverb_only=reverb_only)
    self._reverb_length = reverb_length

  def build(self, audio_shape):
    """Initialize impulse response."""
    initializer = tf.random_normal_initializer(mean=0, stddev=1e-6)
    self.ir = self.add_weight(
        name='ir',
        shape=[self._reverb_length],
        dtype=tf.float32,
        initializer=initializer)

  def get_controls(self, audio):
    """Retrieve ir response."""
    # Match batch dimension.
    ir = self.ir[tf.newaxis, :]
    batch_size = int(audio.shape[0])
    ir = tf.tile(ir, [batch_size, 1])
    controls = {'audio': audio, 'ir': ir}
    return controls


@gin.register
class ExpDecayReverb(Reverb):
  """Parameterize impulse response as a simple exponential decay."""

  def __init__(self,
               reverb_length=64000,
               gain_scale_fn=core.exp_sigmoid,
               reverb_only=False,
               name='exp_decay_reverb'):
    """Constructor.

    Args:
      reverb_length: Length of the impulse response.
      gain_scale_fn: Function by which to scale the network outputs.
      reverb_only: Only output the reverberated signal.
      name: Name of processor module.
    """
    super(ExpDecayReverb, self).__init__(name=name, reverb_only=reverb_only)
    self._reverb_length = reverb_length
    self._gain_scale_fn = gain_scale_fn

  def _get_ir(self, gain, decay):
    """Simple exponential decay of white noise."""
    gain = self._gain_scale_fn(gain)
    decay_exponent = 2.0 + tf.exp(decay)
    time = tf.linspace(0.0, 1.0, self._reverb_length)[tf.newaxis, :]
    noise = tf.random_uniform([1, self._reverb_length], minval=-1.0, maxval=1.0)
    ir = gain * tf.exp(-decay_exponent * time) * noise
    return ir

  def get_controls(self, audio, gain, decay):
    """Convert decoder outputs into ir response.

    Args:
      audio: Dry audio. 2-D Tensor of shape [batch, n_samples].
      gain: Linear gain of impulse response. 2D Tensor of shape [batch, 1].
      decay: Exponential decay coefficient. 2D Tensor of shape [batch, 1].

    Returns:
      controls: Dictionary of effect controls.
    """
    ir = self._get_ir(gain, decay)
    controls = {'audio': audio, 'ir': ir}
    return controls


@gin.register
class TrainableExpDecayReverb(ExpDecayReverb):
  """Parameterize impulse response as a simple exponential decay."""

  def __init__(self,
               reverb_length=64000,
               gain_scale_fn=core.exp_sigmoid,
               reverb_only=False,
               name='exp_decay_reverb'):
    """Constructor.

    Args:
      reverb_length: Length of the impulse response.
      gain_scale_fn: Function by which to scale the network outputs.
      reverb_only: Only output the reverberated signal.
      name: Name of processor module.
    """
    super(TrainableExpDecayReverb, self).__init__(reverb_length=reverb_length,
                                                  gain_scale_fn=gain_scale_fn,
                                                  reverb_only=reverb_only,
                                                  name=name)

  def build(self, audio_shape):
    """Initialize impulse response."""
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

  def get_controls(self, audio):
    """Get parameterized ir response."""
    ir = self._get_ir(self._gain[tf.newaxis, :], self._decay[tf.newaxis, :])
    batch_size = int(audio.shape[0])
    ir = tf.tile(ir, [batch_size, 1])
    controls = {'audio': audio, 'ir': ir}
    return controls
