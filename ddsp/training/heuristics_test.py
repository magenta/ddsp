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

"""Tests for ddsp.training.heuristics."""

from absl.testing import parameterized
from ddsp.training import heuristics
import numpy as np
import tensorflow.compat.v2 as tf


class HeuristicsTest(parameterized.TestCase):

  def test_pad_for_frame(self):
    signal = np.zeros((10,))
    signal[0] = -1
    signal[-1] = 1
    frame_width = 4
    front_padded_signal = heuristics.pad_for_frame(
        signal, mode='front', frame_width=frame_width)
    framed = tf.signal.frame(front_padded_signal, frame_width, 1)
    self.assertEqual(framed.shape[0], signal.shape[0])
    np.testing.assert_array_equal(framed[0], [-1, -1, -1, -1])
    np.testing.assert_array_equal(framed[-1], [0, 0, 0, 1])

    back_padded_signal = heuristics.pad_for_frame(
        signal, mode='end', frame_width=frame_width)
    framed = tf.signal.frame(back_padded_signal, frame_width, 1)
    self.assertEqual(framed.shape[0], signal.shape[0])
    np.testing.assert_array_equal(framed[0], [-1, 0, 0, 0])
    np.testing.assert_array_equal(framed[-1], [1, 1, 1, 1])

    center_padded_signal = heuristics.pad_for_frame(
        signal, mode='center', frame_width=frame_width)
    framed = tf.signal.frame(center_padded_signal, frame_width, 1)
    self.assertEqual(framed.shape[0], signal.shape[0])
    np.testing.assert_array_equal(framed[0], [-1, -1, -1, 0])
    np.testing.assert_array_equal(framed[-1], [0, 0, 1, 1])

  def test_pad_for_frame_odd(self):
    signal = np.zeros((10,))
    signal[0] = -1
    signal[-1] = 1
    frame_width = 3

    center_padded_signal = heuristics.pad_for_frame(
        signal, mode='center', frame_width=frame_width)
    framed = tf.signal.frame(center_padded_signal, frame_width, 1)
    self.assertEqual(framed.shape[0], signal.shape[0])
    np.testing.assert_array_equal(framed[0], [-1, -1, 0])
    np.testing.assert_array_equal(framed[-1], [0, 1, 1])

  @parameterized.parameters(
      (10.0, 2.0, 0.5, (11, 32000)),
      (10.0, 10.0, 0.5, (3, 160000)),
      (10.0, 1.0, 0.5, (21, 16000)),
      (10.0, 1.0, 0.75, (14, 16000)),
  )
  def test_window_array(self, dur, win_len, frame_step_ratio, expected_shape):
    sr = 16000
    sw = np.sin(np.linspace(start=0.0, stop=dur, num=int(sr*dur)))

    wa = heuristics.window_array(sw, sr, win_len, frame_step_ratio)
    self.assertSequenceEqual(wa.shape, expected_shape)
    self.assertEqual(wa.shape[-1], int(sr*win_len))


if __name__ == '__main__':
  tf.test.main()
