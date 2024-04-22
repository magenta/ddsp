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

"""Tests for ddsp.losses."""

from ddsp import core
from ddsp import losses
import numpy as np
import tensorflow as tf


class LossGroupTest(tf.test.TestCase):

  def setUp(self):
    """Create some dummy input data for the chain."""
    super().setUp()

    # Create a network output dictionary.
    self.nn_outputs = {
        'audio': tf.ones((3, 8000), dtype=tf.float32),
        'audio_synth': tf.ones((3, 8000), dtype=tf.float32),
        'magnitudes': tf.ones((3, 200, 2), dtype=tf.float32),
        'f0_hz': 200 + tf.ones((3, 200, 1), dtype=tf.float32),
    }

    # Create Processors.
    spectral_loss = losses.SpectralLoss()
    crepe_loss = losses.PretrainedCREPEEmbeddingLoss(name='crepe_loss')

    # Create DAG for testing.
    self.dag = [
        (spectral_loss, ['audio', 'audio_synth']),
        (crepe_loss, ['audio', 'audio_synth']),
    ]
    self.expected_outputs = [
        'spectral_loss',
        'crepe_loss'
    ]

  def _check_tensor_outputs(self, strings_to_check, outputs):
    for tensor_string in strings_to_check:
      tensor = core.nested_lookup(tensor_string, outputs)
      self.assertIsInstance(tensor, (np.ndarray, tf.Tensor))

  def test_dag_construction(self):
    """Tests if DAG is built properly and runs.
    """
    loss_group = losses.LossGroup(dag=self.dag)
    print('!!!!!!!!!!!', loss_group.dag, loss_group.loss_names, self.dag)
    loss_outputs = loss_group(self.nn_outputs)
    self.assertIsInstance(loss_outputs, dict)
    self._check_tensor_outputs(self.expected_outputs, loss_outputs)


class SpectralLossTest(tf.test.TestCase):

  def test_output_shape_is_correct(self):
    """Test correct shape with all losses active."""
    loss_obj = losses.SpectralLoss(
        mag_weight=1.0,
        delta_time_weight=1.0,
        delta_freq_weight=1.0,
        cumsum_freq_weight=1.0,
        logmag_weight=1.0,
        loudness_weight=1.0,
    )

    input_audio = tf.ones((3, 8000), dtype=tf.float32)
    target_audio = tf.ones((3, 8000), dtype=tf.float32)

    loss = loss_obj(input_audio, target_audio)

    self.assertListEqual([], loss.shape.as_list())
    self.assertTrue(np.isfinite(loss))




class PretrainedCREPEEmbeddingLossTest(tf.test.TestCase):

  def test_output_shape_is_correct(self):
    loss_obj = losses.PretrainedCREPEEmbeddingLoss()

    input_audio = tf.ones((3, 16000), dtype=tf.float32)
    target_audio = tf.ones((3, 16000), dtype=tf.float32)

    loss = loss_obj(input_audio, target_audio)

    self.assertListEqual([], loss.shape.as_list())
    self.assertTrue(np.isfinite(loss))


if __name__ == '__main__':
  tf.test.main()
