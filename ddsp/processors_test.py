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
"""Tests for ddsp.processors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from ddsp import core
from ddsp import effects
from ddsp import processors
from ddsp import synths
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class AddTest(tf.test.TestCase):

  def test_output_is_correct(self):
    processor = processors.Add(name='add')
    x = tf.zeros((2, 3), dtype=tf.float32) + 1.0
    y = tf.zeros((2, 3), dtype=tf.float32) + 2.0
    output = processor(x, y)

    with self.session() as sess:
      actual = sess.run(output)

    expected = np.zeros((2, 3), dtype=np.float32) + 3.0

    self.assertAllEqual(expected, actual)


class SplitTest(tf.test.TestCase):

  def test_output_is_correct(self):
    processor = processors.Split(
        splits=(('x1', 1), ('x2', 2), ('x3', 3)), name='split')
    x1 = np.zeros((2, 3, 1), dtype=np.float32) + 1.0
    x2 = np.zeros((2, 3, 2), dtype=np.float32) + 2.0
    x3 = np.zeros((2, 3, 3), dtype=np.float32) + 3.0
    x = tf.constant(np.concatenate([x1, x2, x3], axis=2))

    output = processor.get_outputs(x)
    call_output = processor(x)
    with self.cached_session() as sess:
      actual = sess.run(output)
      actual_call = sess.run(call_output)

    self.assertSetEqual(set(['split']), set(actual.keys()))

    signal_dict = actual.get('split').get('signal')
    self.assertSetEqual(set(['x1', 'x2', 'x3']), set(signal_dict.keys()))
    self.assertAllEqual(x1, signal_dict.get('x1'))
    self.assertAllEqual(x2, signal_dict.get('x2'))
    self.assertAllEqual(x3, signal_dict.get('x3'))
    self.assertAllEqual(x1, actual_call[0])
    self.assertAllEqual(x2, actual_call[1])
    self.assertAllEqual(x3, actual_call[2])


class MixTest(tf.test.TestCase):

  def test_output_shape_is_correct(self):
    processor = processors.Mix(name='mix')
    x1 = np.zeros((2, 100, 3), dtype=np.float32) + 1.0
    x2 = np.zeros((2, 100, 3), dtype=np.float32) + 2.0
    mix_level = np.zeros(
        (2, 100, 1), dtype=np.float32) + 0.1  # will be passed to sigmoid

    output = processor(x1, x2, mix_level)

    self.assertListEqual([2, 100, 3], output.shape.as_list())


class ProcessorGroupTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    """Create some dummy input data for the chain."""
    super(ProcessorGroupTest, self).setUp()
    # Create inputs.
    self.n_batch = 4
    self.n_frames = 1000
    self.n_time = 64000
    rand_signal = lambda ch: np.random.randn(self.n_batch, self.n_frames, ch)
    nn_outputs = {
        'nn_output': rand_signal(356),
        'f0_hz': 200 + rand_signal(1),
        'target_audio': np.random.randn(self.n_batch, self.n_time)
    }
    self.nn_outputs = {k: core.f32(v) for k, v in nn_outputs.items()}

    # Create Processors.
    additive = synths.Additive(name='additive')
    noise = synths.FilteredNoise(name='noise')
    add = processors.Add(name='add')
    split = processors.Split(
        splits=(('to_amp', 1), ('to_harm', 99), ('to_noise', 256)),
        name='split')
    reverb = effects.FixedReverb(name='reverb')

    # Create DAG for testing.
    dag_tuple = [
        (split, ['nn_output']),
        (additive, ['split/signal/to_amp', 'split/signal/to_harm', 'f0_hz']),
        (noise, ['split/signal/to_noise']),
        (add, ['noise/signal', 'additive/signal']),
        (reverb, ['add/signal']),
    ]
    dag_dict = [dict(processor=t[0], inputs=t[1]) for t in dag_tuple]
    self.dags = {'dag_tuple': dag_tuple, 'dag_dict': dag_dict}

    self.expected_outputs = [
        'nn_output',
        'f0_hz',
        'target_audio',
        'split/signal/to_amp',
        'split/signal/to_harm',
        'split/signal/to_noise',
        'additive/signal',
        'additive/controls/amplitudes',
        'additive/controls/harmonic_distribution',
        'additive/controls/f0_hz',
        'noise/signal',
        'noise/controls/magnitudes',
        'add/signal',
        'reverb/signal',
        'reverb/controls/impulse_response',
        'processor_group/signal',
    ]

  def _check_tensor_outputs(self, strings_to_check, outputs, processor_group):
    for tensor_string in strings_to_check:
      tensor = processor_group._get_input_from_string(tensor_string, outputs)
      self.assertIsInstance(tensor, tf.Tensor)

  @parameterized.named_parameters(
      ('dag_tuple', 'dag_tuple'),
      ('dag_dict', 'dag_dict'),
  )
  def test_dag_construction(self, dag_type):
    """Tests if DAG is built properly and runs.

    Args:
      dag_type: Text name of the type of dag to construct with.
    """
    dag = self.dags[dag_type]
    processor_group = processors.ProcessorGroup(dag=dag, name='processor_group')
    outputs = processor_group.get_outputs(self.nn_outputs)
    self.assertIsInstance(outputs, dict)
    self._check_tensor_outputs(self.expected_outputs, outputs, processor_group)


if __name__ == '__main__':
  tf.test.main()
