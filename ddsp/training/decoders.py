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
"""Library of encoder objects."""

from ddsp.training import nn
import gin
import tensorflow.compat.v2 as tf

tfkl = tf.keras.layers


# ------------------ Decoders --------------------------------------------------
class Decoder(tfkl.Layer):
  """Base class to implement any decoder.

  Users should override decode() to define the actual encoder structure.
  Hyper-parameters will be passed through the constructor.
  """

  def __init__(self,
               output_splits=(('amps', 1), ('harmonic_distribution', 40)),
               name=None):
    super().__init__(name=name)
    self.output_splits = output_splits
    self.n_out = sum([v[1] for v in output_splits])

  def call(self, conditioning):
    """Updates conditioning with dictionary of decoder outputs."""
    x = self.decode(conditioning)
    outputs = nn.split_to_dict(x, self.output_splits)

    if isinstance(outputs, dict):
      conditioning.update(outputs)
    else:
      raise ValueError('Decoder must output a dictionary of signals.')
    return conditioning

  def decode(self, conditioning):
    """Takes in conditioning dictionary, returns dictionary of signals."""
    raise NotImplementedError


@gin.register
class RnnFcDecoder(Decoder):
  """RNN and FC stacks for f0 and loudness."""

  def __init__(self,
               rnn_channels=512,
               rnn_type='gru',
               ch=512,
               layers_per_stack=3,
               input_keys=('ld_scaled', 'f0_scaled', 'z'),
               output_splits=(('amps', 1), ('harmonic_distribution', 40)),
               name=None):
    super().__init__(output_splits=output_splits, name=name)
    stack = lambda: nn.fc_stack(ch, layers_per_stack)
    self.input_keys = input_keys

    # Layers.
    self.input_stacks = [stack() for k in self.input_keys]
    self.rnn = nn.rnn(rnn_channels, rnn_type)
    self.out_stack = stack()
    self.dense_out = nn.dense(self.n_out)

    # Backwards compatability.
    self.f_stack = self.input_stacks[0] if len(self.input_stacks) >= 1 else None
    self.l_stack = self.input_stacks[1] if len(self.input_stacks) >= 2 else None
    self.z_stack = self.input_stacks[2] if len(self.input_stacks) >= 3 else None

  def decode(self, conditioning):
    # Initial processing.
    inputs = [conditioning[k] for k in self.input_keys]
    inputs = [stack(x) for stack, x in zip(self.input_stacks, inputs)]

    # Run an RNN over the latents.
    x = tf.concat(inputs, axis=-1)
    x = self.rnn(x)
    x = tf.concat(inputs + [x], axis=-1)

    # Final processing.
    x = self.out_stack(x)
    return self.dense_out(x)


