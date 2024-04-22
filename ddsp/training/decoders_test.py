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

"""Tests for ddsp.training.decoders."""

import functools

from absl.testing import parameterized
from ddsp.training import decoders
import numpy as np
import tensorflow.compat.v2 as tf


class DilatedConvDecoderTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    """Create some common default values for decoder."""
    super().setUp()
    # For decoder.
    self.ch = 4
    self.layers_per_stack = 3
    self.stacks = 2
    self.output_splits = (('amps', 1), ('harmonic_distribution', 10),
                          ('noise_magnitudes', 10))

    # For audio features and conditioning.
    self.frame_rate = 100
    self.length_in_sec = 0.20
    self.time_steps = int(self.frame_rate * self.length_in_sec)

  def _gen_dummy_conditioning(self):
    """Generate dummy scaled f0 and ld conditioning."""
    conditioning = {}
    # Generate dummy `f0_hz` with batch and channel dims.
    f0_hz_dummy = np.repeat(1.0,
                            self.length_in_sec * self.frame_rate)[np.newaxis, :,
                                                                  np.newaxis]
    conditioning['f0_scaled'] = f0_hz_dummy  # Testing correct shapes only.
    # Generate dummy `loudness_db` with batch and channel dims.
    loudness_db_dummy = np.repeat(1.0, self.length_in_sec *
                                  self.frame_rate)[np.newaxis, :, np.newaxis]
    conditioning[
        'ld_scaled'] = loudness_db_dummy  # Testing correct shapes only.
    return conditioning

  def test_correct_output_splits_and_shapes_dilated_conv_decoder(self):
    decoder = decoders.DilatedConvDecoder(
        ch=self.ch,
        layers_per_stack=self.layers_per_stack,
        stacks=self.stacks,
        conditioning_keys=None,
        output_splits=self.output_splits)

    conditioning = self._gen_dummy_conditioning()
    output = decoder(conditioning)
    for output_name, output_dim in self.output_splits:
      dummy_output = np.zeros((1, self.time_steps, output_dim))
      self.assertShapeEqual(dummy_output, output[output_name])


class RnnFcDecoderTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    """Create some common default values for decoder."""
    super().setUp()
    # For decoder.
    self.input_keys = ('pw_scaled', 'f0_scaled')
    self.output_splits = (('amps', 1), ('harmonic_distribution', 10))
    self.n_batch = 2
    self.n_t = 4
    self.inputs = {
        'pw_scaled': tf.ones([self.n_batch, self.n_t, 1]),
        'f0_scaled': tf.ones([self.n_batch, self.n_t, 1]),
    }
    self.rnn_ch = 3
    self.get_decoder = functools.partial(
        decoders.RnnFcDecoder,
        rnn_channels=self.rnn_ch,
        ch=2,
        layers_per_stack=1,
        input_keys=self.input_keys,
        output_splits=self.output_splits,
        rnn_type='gru',
    )

  @parameterized.named_parameters(
      ('stateful', False),
      ('stateless', True),
  )
  def test_correct_outputs(self, stateless=False):
    decoder = self.get_decoder(stateless=stateless)

    # Add state.
    inputs = self.inputs
    if stateless:
      inputs['state'] = tf.ones([self.n_batch, self.rnn_ch])

    # Run through the network
    outputs = decoder(inputs)

    # Check normal outputs.
    for name, dim in self.output_splits:
      dummy_output = np.zeros((self.n_batch, self.n_t, dim))
      self.assertShapeEqual(dummy_output, outputs[name])

    # Check the explicit state.
    if stateless:
      self.assertShapeEqual(inputs['state'], outputs['state'])
      self.assertNotAllEqual(inputs['state'], outputs['state'])


if __name__ == '__main__':
  tf.test.main()
