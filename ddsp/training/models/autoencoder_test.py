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

"""Tests for ddsp.training.models.autoencoder."""

from absl.testing import parameterized
from ddsp.core import tf_float32
from ddsp.training import models
import gin
import numpy as np
import pkg_resources
import tensorflow as tf

GIN_PATH = pkg_resources.resource_filename(__name__, '../gin')
gin.add_config_file_search_path(GIN_PATH)


class AutoencoderTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('nsynth_ae', 16000, 250, False, 'papers/iclr2020/nsynth_ae.gin'),
      ('solo_instrument', 16000, 250, False,
       'papers/iclr2020/solo_instrument.gin'),
      ('vst_16kHz', 16000, 50, True, 'models/vst/vst.gin'),
      ('vst_32kHz', 32000, 50, True, 'models/vst/vst_32k.gin'),
      ('vst_48kHz', 48000, 50, True, 'models/vst/vst_48k.gin'),
  )
  def test_build_model(self, sample_rate, frame_rate, centered, gin_file):
    """Tests if Model builds properly and produces audio of correct shape.

    Args:
      sample_rate: Sample rate of audio.
      frame_rate: Frame rate of features.
      centered: Add an additional frame.
      gin_file: Name of gin_file to use.
    """
    n_batch = 1
    n_secs = 4
    # frame_size = sample_rate // frame_rate
    n_frames = int(frame_rate * n_secs)
    n_samples = int(sample_rate * n_secs)
    n_samples_16k = int(16000 * n_secs)
    if centered:
      n_frames += 1
      # n_samples += frame_size
      # n_samples_16k += 320

    inputs = {
        'loudness_db': np.zeros([n_batch, n_frames]),
        'f0_hz': np.zeros([n_batch, n_frames]),
        'f0_confidence': np.zeros([n_batch, n_frames]),
        'audio': np.random.randn(n_batch, n_samples),
        'audio_16k': np.random.randn(n_batch, n_samples_16k),
    }
    inputs = {k: tf_float32(v) for k, v in inputs.items()}

    with gin.unlock_config():
      gin.clear_config()
      gin.parse_config_file(gin_file)

    model = models.Autoencoder()
    outputs = model(inputs)
    self.assertIsInstance(outputs, dict)
    # Confirm that model generates correctly sized audio.
    audio_gen = model.get_audio_from_outputs(outputs)
    audio_gen_shape = audio_gen.shape.as_list()
    self.assertEqual(audio_gen_shape, list(inputs['audio'].shape))


if __name__ == '__main__':
  tf.test.main()
