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

from absl.testing import parameterized
from ddsp.core import tf_float32
from ddsp.training import models
import gin
import numpy as np
import pkg_resources
import tensorflow.compat.v2 as tf

GIN_PATH = pkg_resources.resource_filename(__name__, 'gin')
gin.add_config_file_search_path(GIN_PATH)


class AutoencoderTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    """Create some dummy input data for the chain."""
    super().setUp()
    # Create inputs.
    self.n_batch = 4
    self.n_frames = 1001
    self.n_samples = 64000
    inputs = {
        'loudness_db': np.zeros([self.n_batch, self.n_frames]),
        'f0_hz': np.zeros([self.n_batch, self.n_frames]),
        'audio': np.random.randn(self.n_batch, self.n_samples),
    }
    self.inputs = {k: tf_float32(v) for k, v in inputs.items()}

  @parameterized.named_parameters(
      ('nsynth_ae', 'papers/iclr2020/nsynth_ae.gin'),
      ('nsynth_ae_abs', 'papers/iclr2020/nsynth_ae_abs.gin'),
      ('solo_instrument', 'papers/iclr2020/solo_instrument.gin'),
  )
  def test_build_model(self, gin_file):
    """Tests if Model builds properly and produces audio of correct shape.

    Args:
      gin_file: Name of gin_file to use.
    """
    with gin.unlock_config():
      gin.clear_config()
      gin.parse_config_file(gin_file)

    model = models.Autoencoder()
    controls = model.get_controls(self.inputs)
    self.assertIsInstance(controls, dict)
    # Confirm that model generates correctly sized audio.
    audio_gen_shape = controls['processor_group']['signal'].shape.as_list()
    self.assertEqual(audio_gen_shape, list(self.inputs['audio'].shape))

if __name__ == '__main__':
  tf.test.main()
