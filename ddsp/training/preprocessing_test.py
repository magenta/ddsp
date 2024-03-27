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

"""Tests for ddsp.training.preprocessing."""

from absl.testing import parameterized
from ddsp.core import resample
from ddsp.spectral_ops import compute_power
from ddsp.training import preprocessing
import tensorflow as tf

tfkl = tf.keras.layers


class F0PowerPreprocessorTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    """Create input dictionary and preprocessor."""
    super().setUp()
    sr = 16000
    frame_rate = 250
    frame_size = 256
    n_samples = 16000
    n_t = 250
    # Replicate preprocessor computations.
    audio = 0.5 * tf.sin(tf.range(0, n_samples, dtype=tf.float32))[None, :]
    power_db = compute_power(audio,
                             sample_rate=sr,
                             frame_rate=frame_rate,
                             frame_size=frame_size)
    power_db = preprocessing.at_least_3d(power_db)
    power_db = resample(power_db, n_t)
    self.input_dict = {
        'f0_hz': tf.ones([1, n_t]),
        'audio': audio,
        'power_db': power_db,
    }
    self.preprocessor = preprocessing.F0PowerPreprocessor(
        time_steps=n_t,
        frame_rate=frame_rate,
        sample_rate=sr)

  @parameterized.named_parameters(
      ('audio_only', ['audio']),
      ('power_only', ['power_db']),
      ('audio_and_power', ['audio', 'power_db']),
  )
  def test_audio_only(self, input_keys):
    input_keys += ['f0_hz']
    inputs = {k: v for k, v in self.input_dict.items() if k in input_keys}
    outputs = self.preprocessor(inputs)
    self.assertAllClose(self.input_dict['power_db'],
                        outputs['pw_db'],
                        rtol=0.5,
                        atol=30)

if __name__ == '__main__':
  tf.test.main()
