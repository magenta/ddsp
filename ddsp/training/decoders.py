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
"""Library of encoder objects."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ddsp.training import nn
import gin
import tensorflow.compat.v1 as tf

tfkl = tf.keras.layers


# ------------------ Decoders --------------------------------------------------
class Decoder(object):
  """Base class to implement any decoder.

  Users should override decode() to define the actual encoder structure.
  Hyper-parameters will be passed through the constructor.
  """

  def __init__(self, name):
    self.name = name

  def __call__(self, conditioning):
    return self.get_outputs(conditioning)

  def get_outputs(self, conditioning):
    """Updates conditioning with z and (optionally) f0."""
    decoder_outputs = self.decode(conditioning)
    if isinstance(decoder_outputs, dict):
      conditioning.update(decoder_outputs)
    else:
      conditioning['decoder_outputs'] = decoder_outputs
    return conditioning

  def decode(self, conditioning):
    raise NotImplementedError


@gin.configurable
class ZRnnFcDecoder(Decoder):
  """Decompress z in time with RNN. Fully connected stacks for z as well."""

  def __init__(self,
               rnn_channels=512,
               rnn_type='gru',
               ch=512,
               layers_per_stack=3,
               append_f0_loudness=True,
               n_out=41,
               name='z_rnn_fc_decoder'):
    super(ZRnnFcDecoder, self).__init__(name=name)
    self.rnn_channels = rnn_channels
    self.rnn_type = rnn_type
    self.ch = ch
    self.layers_per_stack = layers_per_stack
    self.append_f0_loudness = append_f0_loudness
    self.n_out = n_out

  def decode(self, conditioning):
    rnn = {'lstm': tfkl.LSTM, 'gru': tfkl.GRU}[self.rnn_type]
    stack = lambda x: nn.fc_stack(x, self.ch, self.layers_per_stack)

    f, l, z = conditioning['f0'], conditioning['loudness'], conditioning['z']

    # Initial processing.
    f = stack(f)
    l = stack(l)
    z = stack(z)

    # Expand z with an RNN.
    z = tf.concat([f, l, z], axis=-1) if self.append_f0_loudness else z
    z = rnn(self.rnn_channels, return_sequences=True)(z)
    x = tf.concat([f, l, z], axis=-1)

    # Final processing.
    x = stack(x)
    decoder_outputs = tfkl.Dense(self.n_out)(x)
    return decoder_outputs


@gin.configurable
class RnnFcDecoder(Decoder):
  """RNN and FC stacks for f0 and loudness."""

  def __init__(self,
               rnn_channels=512,
               rnn_type='gru',
               ch=512,
               layers_per_stack=3,
               n_out=41,
               name='rnn_fc_decoder'):
    super(RnnFcDecoder, self).__init__(name=name)
    self.rnn_channels = rnn_channels
    self.rnn_type = rnn_type
    self.ch = ch
    self.layers_per_stack = layers_per_stack
    self.n_out = n_out

  def decode(self, conditioning):
    rnn = {'lstm': tfkl.LSTM, 'gru': tfkl.GRU}[self.rnn_type]
    stack = lambda x: nn.fc_stack(x, self.ch, self.layers_per_stack)

    f, l = conditioning['f0'], conditioning['loudness']

    # Initial processing.
    f = stack(f)
    l = stack(l)

    # Expand z with an RNN.
    z = tf.concat([f, l], axis=-1)
    z = rnn(self.rnn_channels, return_sequences=True)(z)
    x = tf.concat([f, l, z], axis=-1)

    # Final processing.
    x = stack(x)
    decoder_outputs = tfkl.Dense(self.n_out)(x)
    return decoder_outputs


