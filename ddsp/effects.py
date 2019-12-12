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


#------------------ Effects ----------------------------------------------#
@gin.configurable
class Reverb(processors.Processor):
  """Apply convolutional reverb."""

  def __init__(self, scale_fn=None, name='reverb_effect'):
    """Constructor.

    Args:
      scale_fn: A callable to scale the impulse response variable.
      name: Name of processor module.
    """
    super(Reverb, self).__init__(name=name)
    self.scale_fn = scale_fn

  def get_controls(self, nn_outputs, input_audio):
    """Convert decoder outputs into ir response.

    Args:
      nn_outputs: 3-D Tensor  of shape [batch, impulse_response_size, 1].
      input_audio: 2-D Tensor of shape [batch, n_samples].

    Returns:
      controls: Dictionary of effect controls.
    """
    # Scale the amplitudes.
    ir = self.scale_fn(nn_outputs) if self.scale_fn else nn_outputs
    controls = {'input_audio': input_audio, 'impulse_response': ir}
    return controls

  def get_signal(self, input_audio, impulse_response):
    """Apply impulse response.

    Args:
      input_audio: 2-D Tensor of shape [batch, n_samples].
      impulse_response: 3-D Tensor  of shape [batch, impulse_response_size, 1].

    Returns:
      tensor of shape [batch, n_samples]
    """
    return core.fft_convolve(
        input_audio, impulse_response, padding='same', delay_compensation=0)


@gin.configurable
class FixedReverb(Reverb):
  """Apply a fixed convolutional reverb.

  This class creates a tensorflow variable of shape [1, self.reverb_length].

  """

  def __init__(self,
               scale_fn=None,
               reverb_length=64000,
               name='fixed_reverb_effect'):
    """Constructor.

    Args:
      scale_fn: A callable to scale the impulse response variable.
      reverb_length: Length of the impulse response.
      name: Name of processor module.
    """
    super(FixedReverb, self).__init__(scale_fn=scale_fn, name=name)
    self.reverb_length = reverb_length

  def build(self, audio_shape):
    """Initialize impulse response."""
    initializer = tf.random_normal_initializer(mean=0, stddev=1e-6)
    self.ir = self.add_weight(
        name='impulse_response',
        shape=[1, self.reverb_length],
        dtype=tf.float32,
        initializer=initializer)

  def get_controls(self, audio):
    """Retrieve ir response."""

    # Scale the impulse response.
    ir = self.scale_fn(self.ir) if self.scale_fn else self.ir

    # Hold the first impulse response (dry) as zero to decouple dry/wet signal.
    ir = tf.concat([tf.zeros_like(ir, tf.float32)[:, 0:1], ir[:, 1:]], axis=-1)

    # Match batch dimension. Take first output if it's a list.
    batch_size = int(audio.shape[0])
    ir = tf.tile(ir, [batch_size, 1])
    controls = {'input_audio': audio, 'impulse_response': ir}
    return controls
