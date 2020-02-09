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
"""Tests for ddsp.losses."""

from ddsp import losses
import tensorflow.compat.v2 as tf


class SpectralLossTest(tf.test.TestCase):

  def test_output_shape_is_correct(self):
    loss_obj = losses.SpectralLoss()

    input_audio = tf.random.uniform((3, 16000), dtype=tf.float32)
    target_audio = tf.random.uniform((3, 16000), dtype=tf.float32)

    loss = loss_obj(input_audio, target_audio)

    self.assertListEqual([], loss.shape.as_list())


class PretrainedCREPEEmbeddingLossTest(tf.test.TestCase):

  def test_output_shape_is_correct(self):
    loss_obj = losses.PretrainedCREPEEmbeddingLoss()

    input_audio = tf.random.uniform((3, 16000), dtype=tf.float32)
    target_audio = tf.random.uniform((3, 16000), dtype=tf.float32)

    loss = loss_obj(input_audio, target_audio)

    self.assertListEqual([], loss.shape.as_list())


if __name__ == '__main__':
  tf.test.main()
