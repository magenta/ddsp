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

"""Tests for ddsp.synths."""

from ddsp import core
from ddsp import synths
import numpy as np
import tensorflow.compat.v2 as tf


class HarmonicTest(tf.test.TestCase):

  def test_output_shape_is_correct(self):
    synthesizer = synths.Harmonic(
        n_samples=64000,
        sample_rate=16000,
        scale_fn=None,
        normalize_below_nyquist=True)
    batch_size = 3
    num_frames = 1000
    amp = tf.zeros((batch_size, num_frames, 1), dtype=tf.float32) + 1.0
    harmonic_distribution = tf.zeros(
        (batch_size, num_frames, 16), dtype=tf.float32) + 1.0 / 16
    f0_hz = tf.zeros((batch_size, num_frames, 1), dtype=tf.float32) + 16000

    output = synthesizer(amp, harmonic_distribution, f0_hz)

    self.assertAllEqual([batch_size, 64000], output.shape.as_list())


class FilteredNoiseTest(tf.test.TestCase):

  def test_output_shape_is_correct(self):
    synthesizer = synths.FilteredNoise(n_samples=16000)
    filter_bank_magnitudes = tf.zeros((3, 16000, 100), dtype=tf.float32) + 3.0
    output = synthesizer(filter_bank_magnitudes)

    self.assertAllEqual([3, 16000], output.shape.as_list())


class WavetableTest(tf.test.TestCase):

  def test_output_shape_is_correct(self):
    synthesizer = synths.Wavetable(
        n_samples=64000,
        sample_rate=16000,
        scale_fn=None)
    batch_size = 3
    num_frames = 1000
    n_wavetable = 1024
    amp = tf.zeros((batch_size, num_frames, 1), dtype=tf.float32) + 1.0
    wavetables = tf.zeros(
        (batch_size, num_frames, n_wavetable), dtype=tf.float32)
    f0_hz = tf.zeros((batch_size, num_frames, 1), dtype=tf.float32) + 440

    output = synthesizer(amp, wavetables, f0_hz)

    self.assertAllEqual([batch_size, 64000], output.shape.as_list())


class SinusoidalTest(tf.test.TestCase):

  def test_output_shape_is_correct(self):
    synthesizer = synths.Sinusoidal(n_samples=32000, sample_rate=16000)
    batch_size = 3
    num_frames = 1000
    n_partials = 10
    amps = tf.zeros((batch_size, num_frames, n_partials),
                    dtype=tf.float32)
    freqs = tf.zeros((batch_size, num_frames, n_partials),
                     dtype=tf.float32)

    output = synthesizer(amps, freqs)

    self.assertAllEqual([batch_size, 32000], output.shape.as_list())

  def test_frequencies_controls_are_bounded(self):
    depth = 10
    def freq_scale_fn(x):
      return core.frequencies_sigmoid(x, depth=depth, hz_min=0.0, hz_max=8000.0)

    synthesizer = synths.Sinusoidal(
        n_samples=32000, sample_rate=16000, freq_scale_fn=freq_scale_fn)
    batch_size = 3
    num_frames = 10
    n_partials = 100
    amps = tf.zeros((batch_size, num_frames, n_partials), dtype=tf.float32)
    freqs = tf.linspace(-100.0, 100.0, n_partials)
    freqs = tf.tile(freqs[tf.newaxis, tf.newaxis, :, tf.newaxis],
                    [batch_size, num_frames, 1, depth])

    controls = synthesizer.get_controls(amps, freqs)
    freqs = controls['frequencies']
    lt_nyquist = (freqs <= 8000.0)
    gt_zero = (freqs >= 0.0)
    both_conditions = np.logical_and(lt_nyquist, gt_zero)

    self.assertTrue(np.all(both_conditions))


if __name__ == '__main__':
  tf.test.main()
