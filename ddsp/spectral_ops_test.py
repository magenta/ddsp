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

from ddsp import spectral_ops
import numpy as np
import tensorflow.compat.v2 as tf


class STFTTest(tf.test.TestCase):

  def test_tf_and_np_are_consistent(self):
    amp = 1e-2
    audio = amp * (np.random.rand(64000).astype(np.float32) * 2.0 - 1.0)
    frame_size = 2048
    hop_size = 128
    overlap = 1.0 - float(hop_size) / frame_size
    pad_end = True

    s_np = spectral_ops.stft_np(
        audio, frame_size=frame_size, overlap=overlap, pad_end=pad_end)

    s_tf = spectral_ops.stft(
        audio, frame_size=frame_size, overlap=overlap, pad_end=pad_end)

    # TODO(jesseengel): The phase comes out a little different, figure out why.
    self.assertAllClose(np.abs(s_np), np.abs(s_tf), rtol=1e-3, atol=1e-3)


class LoudnessTest(tf.test.TestCase):

  def test_tf_and_np_are_consistent(self):
    amp = 1e-2
    audio = amp * (np.random.rand(64000).astype(np.float32) * 2.0 - 1.0)
    frame_size = 2048
    frame_rate = 250

    ld_tf = spectral_ops.compute_loudness(
        audio, n_fft=frame_size, frame_rate=frame_rate, use_tf=True)

    ld_np = spectral_ops.compute_loudness(
        audio, n_fft=frame_size, frame_rate=frame_rate, use_tf=False)

    self.assertAllClose(np.abs(ld_np), np.abs(ld_tf), rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
  tf.test.main()
