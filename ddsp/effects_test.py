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

"""Tests for ddsp.effects."""

from absl.testing import parameterized
from ddsp import effects
import tensorflow.compat.v2 as tf


class ReverbTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    """Creates some test specific attributes."""
    super().setUp()
    self.reverb_class = effects.Reverb
    self.audio = tf.zeros((3, 16000))
    self.construct_args = {'reverb_length': 100}
    self.call_args = {'ir': tf.zeros((3, 100, 1))}
    self.controls_keys = ['audio', 'ir']

  @parameterized.named_parameters(
      ('trainable', True),
      ('not_trainable', False),
  )
  def test_output_shape_and_variables_are_correct(self, trainable):
    reverb = self.reverb_class(trainable=trainable, **self.construct_args)
    if trainable:
      output = reverb(self.audio)
    else:
      output = reverb(self.audio, **self.call_args)

    self.assertListEqual(list(self.audio.shape), output.shape.as_list())
    self.assertEqual(reverb.trainable, trainable)
    self.assertEmpty(reverb.non_trainable_variables)
    assert_variables = self.assertNotEmpty if trainable else self.assertEmpty
    assert_variables(reverb.trainable_variables)

  def test_non_trainable_raises_value_error(self):
    reverb = self.reverb_class(trainable=False, **self.construct_args)
    with self.assertRaises(ValueError):
      _ = reverb(self.audio)

  @parameterized.named_parameters(
      ('trainable', True),
      ('not_trainable', False),
  )
  def test_get_controls_returns_correct_keys(self, trainable):
    reverb = self.reverb_class(trainable=trainable, **self.construct_args)
    reverb.build(self.audio.shape)
    if trainable:
      controls = reverb.get_controls(self.audio)
    else:
      controls = reverb.get_controls(self.audio, **self.call_args)

    self.assertListEqual(list(controls.keys()), self.controls_keys)


class ExpDecayReverbTest(ReverbTest):

  def setUp(self):
    """Creates some test specific attributes."""
    super().setUp()
    self.reverb_class = effects.ExpDecayReverb
    self.audio = tf.zeros((3, 16000))
    self.construct_args = {'reverb_length': 100}
    self.call_args = {'gain': tf.zeros((3, 1)),
                      'decay': tf.zeros((3, 1))}


class FilteredNoiseReverbTest(ReverbTest):

  def setUp(self):
    """Creates some test specific attributes."""
    super().setUp()
    self.reverb_class = effects.FilteredNoiseReverb
    self.audio = tf.zeros((3, 16000))
    self.construct_args = {'reverb_length': 100,
                           'n_frames': 10,
                           'n_filter_banks': 20}
    self.call_args = {'magnitudes': tf.zeros((3, 10, 20))}


class FIRFilterTest(tf.test.TestCase):

  def test_output_shape_is_correct(self):
    processor = effects.FIRFilter()

    audio = tf.zeros((3, 16000))
    magnitudes = tf.zeros((3, 100, 30))
    output = processor(audio, magnitudes)

    self.assertListEqual([3, 16000], output.shape.as_list())


class ModDelayTest(tf.test.TestCase):

  def test_output_shape_is_correct(self):
    processor = effects.ModDelay()

    audio = tf.zeros((3, 16000))
    gain = tf.zeros((3, 16000, 1))
    phase = tf.zeros((3, 16000, 1))
    output = processor(audio, gain, phase)

    self.assertListEqual([3, 16000], output.shape.as_list())


if __name__ == '__main__':
  tf.test.main()
