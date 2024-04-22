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

"""Library of helper functions for testing."""

import numpy as np


def gen_np_sinusoid(frequency, amp, sample_rate, audio_len_sec):
  x = np.linspace(0, audio_len_sec, int(audio_len_sec * sample_rate))
  audio_sin = amp * (np.sin(2 * np.pi * frequency * x))
  return audio_sin


def gen_np_batched_sinusoids(frequency, amp, sample_rate, audio_len_sec,
                             batch_size):
  batch_sinusoids = [
      gen_np_sinusoid(frequency, amp, sample_rate, audio_len_sec)
      for _ in range(batch_size)
  ]
  return np.array(batch_sinusoids)

