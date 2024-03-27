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

"""Library of decoder layers."""

from ddsp import core
from ddsp.training import nn
import gin
import tensorflow as tf

tfkl = tf.keras.layers


# ------------------ Decoders --------------------------------------------------
@gin.register
class RnnFcDecoder(nn.DictLayer):
  """RNN and FC stacks for f0 and loudness."""

  def __init__(self,
               rnn_channels=512,
               rnn_type='gru',
               ch=512,
               layers_per_stack=3,
               stateless=False,
               input_keys=('ld_scaled', 'f0_scaled', 'z'),
               output_splits=(('amps', 1), ('harmonic_distribution', 40)),
               **kwargs):
    """Constructor.

    Args:
      rnn_channels: Dims for the RNN layer.
      rnn_type: Either 'gru' or 'lstm'.
      ch: Dims of the fully connected layers.
      layers_per_stack: Fully connected layers per a stack.
      stateless: Change api to explicitly pass in and out RNN state. Needed for
        SavedModel/TFLite inference. Uses nn.StatelessRnn.
      input_keys: Create a fully connected stack for each input.
      output_splits: Splits the outputs into these dimensions.
      **kwargs: Keras-specific kwargs.

    Returns:
      Dictionary with keys from output_splits. Also has 'state' key if
        `stateless=True`, for manually handling state.
    """
    # Manually handle state if stateless.
    self.stateless = stateless

    # Always put state as the last input and output.
    self.output_splits = output_splits
    output_keys = [v[0] for v in output_splits]
    if self.stateless:
      input_keys = list(input_keys) + ['state']
      output_keys = list(output_keys) + ['state']

    super().__init__(input_keys=input_keys, output_keys=output_keys, **kwargs)

    # Don't create a stack for manual RNN state.
    stack = lambda: nn.FcStack(ch, layers_per_stack)
    n_stacks = len(self.input_keys)
    if self.stateless:
      n_stacks -= 1
    rnn_cls = nn.StatelessRnn if stateless else nn.Rnn

    # Layers.
    self.input_stacks = [stack() for _ in range(n_stacks)]
    self.rnn = rnn_cls(rnn_channels, rnn_type)
    self.out_stack = stack()

    # Copied from OutputSplitsLayer to handle stateless logic.
    n_out = sum([v[1] for v in output_splits])
    self.dense_out = tfkl.Dense(n_out)

  def call(self, *inputs, **unused_kwargs):
    # Last input is always carried state for stateless RNN.
    inputs = list(inputs)
    if self.stateless:
      state = inputs.pop()

    # Initial processing.
    inputs = [stack(x) for stack, x in zip(self.input_stacks, inputs)]

    # Run an RNN over the latents.
    x = tf.concat(inputs, axis=-1)
    if self.stateless:
      x, new_state = self.rnn(x, state)
    else:
      x = self.rnn(x)
    x = tf.concat(inputs + [x], axis=-1)

    # Final processing.
    x = self.out_stack(x)
    x = self.dense_out(x)

    output_dict = nn.split_to_dict(x, self.output_splits)
    if self.stateless:
      output_dict['state'] = new_state

    return output_dict


@gin.register
class MidiDecoder(nn.DictLayer):
  """Decodes MIDI notes (& velocities) to f0 (& loudness)."""

  def __init__(self,
               net=None,
               f0_residual=True,
               center_loudness=True,
               norm=True,
               **kwargs):
    """Constructor."""
    super().__init__(**kwargs)
    self.net = net
    self.f0_residual = f0_residual
    self.center_loudness = center_loudness
    self.dense_out = tfkl.Dense(2)
    self.norm = nn.Normalize('layer') if norm else None

  def call(self, z_pitch, z_vel=None, z=None) -> ['f0_midi', 'loudness']:
    """Forward pass for the MIDI decoder.

    Args:
      z_pitch: Tensor containing encoded pitch in MIDI scale. [batch, time, 1].
      z_vel: Tensor containing encoded velocity in MIDI scale. [batch, time, 1].
      z: Additional non-MIDI latent tensor. [batch, time, n_z]

    Returns:
      f0_midi, loudness: Reconstructed f0 and loudness.
    """
    # pylint: disable=unused-argument
    # x = tf.concat([z_pitch, z_vel], axis=-1)  # TODO(jesse): Allow velocity.
    x = z_pitch
    x = self.net(x) if z is None else self.net([x, z])

    if self.norm is not None:
      x = self.norm(x)

    x = self.dense_out(x)

    f0_midi = x[..., 0:1]
    loudness = x[..., 1:2]

    if self.f0_residual:
      f0_midi += z_pitch

    if self.center_loudness:
      loudness = loudness * 30.0 - 70.0

    return f0_midi, loudness


@gin.register
class MidiToHarmonicDecoder(nn.DictLayer):
  """Decodes MIDI notes (& velocities) to f0, amps, hd, noise."""

  def __init__(self,
               net=None,
               f0_residual=True,
               norm=True,
               output_splits=(('f0_midi', 1),
                              ('amplitudes', 1),
                              ('harmonic_distribution', 60),
                              ('magnitudes', 65)),
               midi_zero_silence=True,
               **kwargs):
    """Constructor."""
    self.output_splits = output_splits
    self.n_out = sum([v[1] for v in output_splits])
    output_keys = [v[0] for v in output_splits] + ['f0_hz']
    super().__init__(output_keys=output_keys, **kwargs)

    # Layers.
    self.net = net
    self.f0_residual = f0_residual
    self.dense_out = tfkl.Dense(self.n_out)
    self.norm = nn.Normalize('layer') if norm else None
    self.midi_zero_silence = midi_zero_silence

  def call(self, z_pitch, z_vel=None, z=None):
    """Forward pass for the MIDI decoder.

    Args:
      z_pitch: Tensor containing encoded pitch in MIDI scale. [batch, time, 1].
      z_vel: Tensor containing encoded velocity in MIDI scale. [batch, time, 1].
      z: Additional non-MIDI latent tensor. [batch, time, n_z]

    Returns:
      A dictionary to feed into a processor group.
    """
    # pylint: disable=unused-argument
    # x = tf.concat([z_pitch, z_vel], axis=-1)  # TODO(jesse): Allow velocity.
    x = z_pitch
    x = self.net(x) if z is None else self.net([x, z])

    if self.norm is not None:
      x = self.norm(x)

    x = self.dense_out(x)

    outputs = nn.split_to_dict(x, self.output_splits)

    if self.f0_residual:
      outputs['f0_midi'] += z_pitch

    outputs['f0_hz'] = core.midi_to_hz(outputs['f0_midi'],
                                       midi_zero_silence=self.midi_zero_silence)
    return outputs


@gin.register
class DilatedConvDecoder(nn.OutputSplitsLayer):
  """WaveNet style 1-D dilated convolution with optional conditioning."""

  def __init__(self,
               ch=256,
               kernel_size=3,
               layers_per_stack=5,
               stacks=2,
               dilation=2,
               norm_type='layer',
               resample_stride=1,
               stacks_per_resample=1,
               resample_after_convolve=True,
               input_keys=('ld_scaled', 'f0_scaled'),
               output_splits=(('amps', 1), ('harmonic_distribution', 60)),
               conditioning_keys=('z'),
               precondition_stack=None,
               spectral_norm=False,
               ortho_init=False,
               **kwargs):
    """Constructor, combines input_keys and conditioning_keys."""
    self.conditioning_keys = ([] if conditioning_keys is None else
                              list(conditioning_keys))
    input_keys = list(input_keys) + self.conditioning_keys
    super().__init__(input_keys, output_splits, **kwargs)

    # Conditioning.
    self.n_conditioning = len(self.conditioning_keys)
    self.conditional = bool(self.conditioning_keys)
    if not self.conditional and precondition_stack is not None:
      raise ValueError('You must specify conditioning keys if you specify'
                       'a precondition stack.')

    # Layers.
    self.precondition_stack = precondition_stack
    self.dilated_conv_stack = nn.DilatedConvStack(
        ch=ch,
        kernel_size=kernel_size,
        layers_per_stack=layers_per_stack,
        stacks=stacks,
        dilation=dilation,
        norm_type=norm_type,
        resample_type='upsample' if resample_stride > 1 else None,
        resample_stride=resample_stride,
        stacks_per_resample=stacks_per_resample,
        resample_after_convolve=resample_after_convolve,
        conditional=self.conditional,
        spectral_norm=spectral_norm,
        ortho_init=ortho_init)

  def _parse_inputs(self, inputs):
    """Split x and z inputs and run preconditioning."""
    if self.conditional:
      x = tf.concat(inputs[:-self.n_conditioning], axis=-1)
      z = tf.concat(inputs[-self.n_conditioning:], axis=-1)
      if self.precondition_stack is not None:
        z = self.precondition_stack(z)
      return [x, z]
    else:
      return tf.concat(inputs, axis=-1)

  def compute_output(self, *inputs):
    stack_inputs = self._parse_inputs(inputs)
    return self.dilated_conv_stack(stack_inputs)


