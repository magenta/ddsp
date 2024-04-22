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

"""Tests for ddsp.training.nn."""

from absl.testing import parameterized
from ddsp.training import nn
import numpy as np
import tensorflow.compat.v2 as tf

tfkl = tf.keras.layers


class DictLayerTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    """Create Class definitions for different ways of writing DictLayer."""
    super().setUp()
    self.x = tf.ones([2, 5])
    n_out = 10
    self.n_out = n_out

    class TestLayer(nn.DictLayer):
      """Explicitly sets input/output keys."""

      def __init__(self,
                   input_keys=('x1', 'x2'),
                   output_keys=('y1', 'y2', 'y3'),
                   **kwargs):
        super().__init__(input_keys, output_keys, **kwargs)
        self.dense = tfkl.Dense(n_out)

      def call(self, x1, x2):
        y1, y2, y3 = self.dense(x1), self.dense(x2), self.dense(x1)
        return y1, y2, y3

    self.TestLayer = TestLayer  # pylint: disable=invalid-name

    class TestLayerAnnotated(nn.DictLayer):
      """Uses args and return annotations to define input/output keys."""

      def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = tfkl.Dense(n_out)

      def call(self, x1, x2) -> ['y1', 'y2', 'y3']:
        y1, y2, y3 = self.dense(x1), self.dense(x2), self.dense(x1)
        return y1, y2, y3

    self.TestLayerAnnotated = TestLayerAnnotated  # pylint: disable=invalid-name

    class TestLayerDefaults(nn.DictLayer):
      """Uses args and return annotations, and default arg inputs."""

      def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = tfkl.Dense(n_out)

      def call(self, x1, x2=None) -> ['y1', 'y2', 'y3']:
        y1, y3 = self.dense(x1), self.dense(x1)
        y2 = self.dense(x2) if x2 is not None else self.dense(x1)
        return y1, y2, y3

    self.TestLayerDefaults = TestLayerDefaults  # pylint: disable=invalid-name

    class TestLayerOptionals(nn.DictLayer):
      """Uses args and return annotations, and default arg inputs."""

      def call(self, x1, x2=None, x3=None) -> ['y']:
        """Optionally add multiples of each input."""
        y = x1
        if x2 is not None:
          y += 2.0 * x2
        if x3 is not None:
          y += 3.0 * x3
        return y

    self.TestLayerOptionals = TestLayerOptionals  # pylint: disable=invalid-name

  def assert_output_shapes_are_correct(self, dict_layer, outputs):
    """Check that the output is correct for a input."""
    self.assertListEqual(list(dict_layer.output_keys), list(outputs.keys()))
    for v in outputs.values():
      self.assertEqual(v.shape[-1], self.n_out)

  @parameterized.named_parameters(
      ('explicit', 'TestLayer'),
      ('args_and_return_annotations', 'TestLayerAnnotated'),
  )
  def test_output_is_correct(self, layer_class):
    """Check that call() args and return annotations give input/output keys.

    Args:
      layer_class: Which type of class definition to use for test_layer.
    """
    test_layer = getattr(self, layer_class)()
    # Arg inputs.
    outputs = test_layer(self.x, self.x)
    self.assert_output_shapes_are_correct(test_layer, outputs)

    # Dict inputs.
    outputs = test_layer({'x1': self.x, 'x2': self.x})
    self.assert_output_shapes_are_correct(test_layer, outputs)

    # Kwarg inputs.
    outputs = test_layer(x1=self.x, x2=self.x)
    self.assert_output_shapes_are_correct(test_layer, outputs)

    # Merge multiple dict inputs, ignore extra keys.
    outputs = test_layer({'x1': self.x, 'ignore': 0},
                         {'x2': self.x, 'ignore2': 0})
    self.assert_output_shapes_are_correct(test_layer, outputs)

    # Raises errors for bad inputs.
    # Wrong number of args.
    with self.assertRaises(TypeError):
      test_layer(self.x)
    # Missing key --> Wrong number of args.
    with self.assertRaises(TypeError):
      test_layer({'x1': self.x})
    # Missing key, wrong key name --> Wrong number of args.
    with self.assertRaises(TypeError):
      test_layer({'asdf': self.x, 'x2': self.x})
    # Duplicate input argument.
    with self.assertRaises(TypeError):
      test_layer(self.x, {'x1': self.x, 'x2': self.x})

  def test_input_output_keys_are_correct(self):
    """Ensure input output keys are the same for different class definitions."""
    layer = self.TestLayer()
    layer_annotated = self.TestLayerAnnotated()
    self.assertAllEqual(layer.input_keys, layer_annotated.input_keys)
    self.assertAllEqual(layer.output_keys, layer_annotated.output_keys)

  @parameterized.named_parameters(
      ('explicit', 'TestLayer'),
      ('args_and_return_annotations', 'TestLayerAnnotated'),
      ('default_args', 'TestLayerDefaults'),
  )
  def test_renaming_input_output_keys(self, layer_class):
    """Ensure input output keys are overwritten by constructor.

    Args:
      layer_class: Which type of class definition to use for test_layer.
    """
    input_keys = ['input_1', 'input_2']
    output_keys = ['output_1', 'output_2', 'output_3']
    test_layer = getattr(self, layer_class)(input_keys=input_keys,
                                            output_keys=output_keys)
    # Arg inputs.
    outputs = test_layer(self.x, self.x)
    self.assert_output_shapes_are_correct(test_layer, outputs)

    # Dict inputs.
    outputs = test_layer({'input_1': self.x, 'input_2': self.x})
    self.assert_output_shapes_are_correct(test_layer, outputs)

    # Make sure original input_keys no longer are correct.
    with self.assertRaises(TypeError):
      test_layer({'x1': self.x, 'x2': self.x})

  def assertOutputsEqual(self, outputs_1, outputs_2):
    """Make sure two output dictionaries are the same."""
    self.assertSetEqual(set(outputs_1.keys()), set(outputs_2.keys()))
    for key in outputs_1.keys():
      self.assertAllClose(outputs_1[key], outputs_2[key])

  def test_default_args_have_correct_outputs(self):
    """Check correct result regardless of whether default arg is provided."""
    test_layer = self.TestLayerDefaults()
    # Arg inputs.
    outputs_1 = test_layer(self.x, self.x)
    outputs_2 = test_layer(self.x)
    self.assertOutputsEqual(outputs_1, outputs_2)

    # Dict inputs.
    outputs_1 = test_layer({'x1': self.x, 'x2': self.x})
    outputs_2 = test_layer({'x1': self.x})
    self.assertOutputsEqual(outputs_1, outputs_2)

  def test_optional_args_read_in_correct_order(self):
    """Check correct result regardless of whether default args are provided."""
    test_layer = self.TestLayerOptionals()
    # Dict inputs.
    outputs_1 = test_layer({'x1': self.x})['y']
    outputs_1_2 = test_layer({'x1': self.x, 'x2': self.x})['y']
    outputs_1_3 = test_layer({'x1': self.x, 'x3': self.x})['y']
    outputs_1_2_3 = test_layer({'x1': self.x, 'x2': self.x, 'x3': self.x})['y']
    self.assertAllClose(outputs_1, self.x)
    self.assertAllClose(outputs_1_2, 3.0 * self.x)
    self.assertAllClose(outputs_1_3, 4.0 * self.x)
    self.assertAllClose(outputs_1_2_3, 6.0 * self.x)


class SplitToDictTest(tf.test.TestCase):

  def test_output_is_correct(self):
    tensor_splits = (('x1', 1), ('x2', 2), ('x3', 3))
    x1 = np.zeros((2, 3, 1), dtype=np.float32) + 1.0
    x2 = np.zeros((2, 3, 2), dtype=np.float32) + 2.0
    x3 = np.zeros((2, 3, 3), dtype=np.float32) + 3.0
    x = tf.constant(np.concatenate([x1, x2, x3], axis=2))

    output = nn.split_to_dict(x, tensor_splits)

    self.assertSetEqual(set(['x1', 'x2', 'x3']), set(output.keys()))
    self.assertAllEqual(x1, output.get('x1'))
    self.assertAllEqual(x2, output.get('x2'))
    self.assertAllEqual(x3, output.get('x3'))


class PolyphaseResampleTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('even_multiple', 4, 2),
      ('stride_n_t', 5, 5),
      ('stride_1', 4, 1),
  )
  def test_reconstructions_are_lossless(self, n_t, stride):
    n_batch = 2
    n_ch = 3
    x = tf.tile(tf.reshape(tf.range(n_t), [1, n_t, 1]), [n_batch, 1, n_ch])
    x_down = nn.polyphase_resample(x, stride, 'down', trim_or_pad='pad')
    x_recon = nn.polyphase_resample(x_down, stride, 'up', trim_or_pad='pad')
    self.assertAllEqual(x, x_recon)


if __name__ == '__main__':
  tf.test.main()
