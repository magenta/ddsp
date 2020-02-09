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
"""Tests for ddsp.training.nn."""

from ddsp.training import nn
import numpy as np
import tensorflow.compat.v2 as tf


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


if __name__ == '__main__':
  tf.test.main()
