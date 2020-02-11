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
"""Tests for ddsp.processors."""

from absl.testing import parameterized
from ddsp import core
from ddsp import effects
from ddsp import processors
from ddsp import synths
import numpy as np
import tensorflow.compat.v2 as tf


class ProcessorGroupTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    """Create some dummy input data for the chain."""
    super().setUp()
    # Create inputs.
    self.n_batch = 4
    self.n_frames = 1000
    self.n_time = 64000
    rand_signal = lambda ch: np.random.randn(self.n_batch, self.n_frames, ch)
    self.nn_outputs = {
        'amps': rand_signal(1),
        'harmonic_distribution': rand_signal(99),
        'magnitudes': rand_signal(256),
        'f0_hz': 200 + rand_signal(1),
        'target_audio': np.random.randn(self.n_batch, self.n_time)
    }

    # Create Processors.
    additive = synths.Additive(name='additive')
    noise = synths.FilteredNoise(name='noise')
    add = processors.Add(name='add')
    reverb = effects.Reverb(trainable=True, name='reverb')

    # Create DAG for testing.
    self.dag = [
        (additive, ['amps', 'harmonic_distribution', 'f0_hz']),
        (noise, ['magnitudes']),
        (add, ['noise/signal', 'additive/signal']),
        (reverb, ['add/signal']),
    ]
    self.expected_outputs = [
        'amps',
        'harmonic_distribution',
        'magnitudes',
        'f0_hz',
        'target_audio',
        'additive/signal',
        'additive/controls/amplitudes',
        'additive/controls/harmonic_distribution',
        'additive/controls/f0_hz',
        'noise/signal',
        'noise/controls/magnitudes',
        'add/signal',
        'reverb/signal',
        'reverb/controls/ir',
        'processor_group/signal',
    ]

  def _check_tensor_outputs(self, strings_to_check, outputs):
    for tensor_string in strings_to_check:
      tensor = core.nested_lookup(tensor_string, outputs)
      self.assertIsInstance(tensor, (np.ndarray, tf.Tensor))

  def test_dag_construction(self):
    """Tests if DAG is built properly and runs.
    """
    processor_group = processors.ProcessorGroup(dag=self.dag,
                                                name='processor_group')
    outputs = processor_group.get_controls(self.nn_outputs)
    self.assertIsInstance(outputs, dict)
    self._check_tensor_outputs(self.expected_outputs, outputs)


class AddTest(tf.test.TestCase):

  def test_output_is_correct(self):
    processor = processors.Add(name='add')
    x = tf.zeros((2, 3), dtype=tf.float32) + 1.0
    y = tf.zeros((2, 3), dtype=tf.float32) + 2.0

    output = processor(x, y)

    expected = np.zeros((2, 3), dtype=np.float32) + 3.0
    self.assertAllEqual(expected, output)


class MixTest(tf.test.TestCase):

  def test_output_shape_is_correct(self):
    processor = processors.Mix(name='mix')
    x1 = np.zeros((2, 100, 3), dtype=np.float32) + 1.0
    x2 = np.zeros((2, 100, 3), dtype=np.float32) + 2.0
    mix_level = np.zeros(
        (2, 100, 1), dtype=np.float32) + 0.1  # will be passed to sigmoid

    output = processor(x1, x2, mix_level)

    self.assertListEqual([2, 100, 3], output.shape.as_list())


if __name__ == '__main__':
  tf.test.main()
