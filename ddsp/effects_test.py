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
"""Tests for ddsp.effects."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ddsp import effects
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class ReverbTest(tf.test.TestCase):

  def test_output_shape_is_correct(self):
    processor = effects.Reverb()

    impulse_response = tf.zeros((3, 100, 1), dtype=tf.float32)
    input_audio = tf.zeros((3, 16000), dtype=tf.float32)

    output = processor(impulse_response, input_audio)

    self.assertListEqual([3, 16000], output.shape.as_list())


class FixedReverbTest(tf.test.TestCase):

  def test_output_shape_is_correct(self):
    processor = effects.FixedReverb(reverb_length=100)

    input_audio = tf.zeros((3, 16000), dtype=tf.float32)

    output = processor(input_audio)

    self.assertListEqual([3, 16000], output.shape.as_list())


if __name__ == '__main__':
  tf.test.main()
