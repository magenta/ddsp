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

    audio = tf.zeros((3, 16000), dtype=tf.float32)
    ir = tf.zeros((3, 100, 1), dtype=tf.float32)

    output = processor(audio, ir)

    self.assertListEqual([3, 16000], output.shape.as_list())


class TrainableReverbTest(tf.test.TestCase):

  def test_output_shape_is_correct(self):
    processor = effects.TrainableReverb(reverb_length=100)

    audio = tf.zeros((3, 16000), dtype=tf.float32)

    output = processor(audio)

    self.assertListEqual([3, 16000], output.shape.as_list())


class ExpDecayReverbTest(tf.test.TestCase):

  def test_output_shape_is_correct(self):
    processor = effects.ExpDecayReverb(reverb_length=100)

    audio = tf.zeros((3, 16000), dtype=tf.float32)
    gain = tf.zeros((3, 1), dtype=tf.float32)
    decay = tf.zeros((3, 1), dtype=tf.float32)

    output = processor(audio, gain, decay)

    self.assertListEqual([3, 16000], output.shape.as_list())


class TrainableExpDecayReverbTest(tf.test.TestCase):

  def test_output_shape_is_correct(self):
    processor = effects.TrainableExpDecayReverb(reverb_length=100)

    audio = tf.zeros((3, 16000), dtype=tf.float32)

    output = processor(audio)

    self.assertListEqual([3, 16000], output.shape.as_list())


if __name__ == '__main__':
  tf.test.main()
