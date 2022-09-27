# Copyright 2022 The DDSP Authors.
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

"""Tests for ddsp.training.data_preparation.prepare_urmp_dataset_lib."""

from absl.testing import absltest
from absl.testing import parameterized

from ddsp.training.data_preparation import prepare_urmp_dataset_lib
import numpy as np


class PrepareUrmpDatasetLibTest(parameterized.TestCase):

  def test_resample(self):
    times_with_offset = np.linspace(0.0, 4.0, 400, endpoint=False) + 0.033
    # ... 3.993, 4.003, 4.013, 4.023
    times_with_offset = times_with_offset[:-2]
    ex = {
        'f0_hz': np.zeros_like(times_with_offset),
        'f0_time': times_with_offset,
        'note_active_frame_indices': np.zeros((1002, 128)),
        'note_active_velocities': np.zeros((1003, 128)),
        'note_onsets': np.zeros((1003, 128)),
        'note_offsets': np.zeros((1003, 128)),
        'audio': np.zeros(64020)
    }
    resampled_ex = prepare_urmp_dataset_lib.resample(
        ex, ddsp_sample_rate=250, audio_sample_rate=16000)
    self.assertEqual(resampled_ex['f0_hz'].shape[0], 1000)
    self.assertEqual(resampled_ex['note_active_velocities'].shape[0], 1000)
    self.assertEqual(resampled_ex['note_active_velocities'].shape[1], 128)
    self.assertEqual(resampled_ex['note_onsets'].shape[0], 1000)
    self.assertEqual(resampled_ex['note_offsets'].shape[0], 1000)
    self.assertEqual(resampled_ex['audio'].shape[0], 64000)

  def test_batched(self):
    ex = {
        'f0_hz': np.zeros(1000),
        'note_active_velocities': np.zeros((1000, 128)),
        'audio': np.zeros(64000)
    }
    batched_ex = prepare_urmp_dataset_lib.batch_dataset(
        ex, audio_sample_rate=16000, ddsp_sample_rate=250)
    self.assertLen(batched_ex, 7)  # 750 frames of silence on each side


if __name__ == '__main__':
  absltest.main()
