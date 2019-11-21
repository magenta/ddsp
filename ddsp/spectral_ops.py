# Copyright 2019 The DDSP Authors.
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
"""Library of FFT operations for loss functions and conditioning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import librosa
import tensorflow.compat.v1 as tf


def safe_log(x, eps=1e-5):
  return tf.log(x + eps)


def stft(audio, frame_size=2048, overlap=0.75, pad_end=False):
  assert frame_size * overlap % 2.0 == 0.0
  s = tf.signal.stft(
      signals=audio,
      frame_length=int(frame_size),
      frame_step=int(frame_size * (1.0 - overlap)),
      fft_length=int(frame_size),
      pad_end=pad_end)
  return s


@gin.configurable
def calc_mag(audio, size=2048, overlap=0.75, pad_end=False):
  return tf.abs(stft(audio, frame_size=size, overlap=overlap, pad_end=pad_end))


@gin.configurable
def calc_mel(audio,
             lo_hz=0.0,
             hi_hz=8000.0,
             bins=64,
             fft_size=2048,
             overlap=0.75,
             pad_end=False):
  """Calculate Mel Spectrogram."""
  mag = calc_mag(audio, fft_size, overlap, pad_end)
  num_spectrogram_bins = mag.shape[-1].value
  linear_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
      bins, num_spectrogram_bins, 16000, lo_hz, hi_hz)
  mel = tf.tensordot(mag, linear_to_mel_matrix, 1)
  mel.set_shape(mag.shape[:-1].concatenate(linear_to_mel_matrix.shape[-1:]))
  return mel


@gin.configurable
def calc_logmag(audio, size=2048):
  return safe_log(calc_mag(audio, size))


@gin.configurable
def calc_logmel(audio,
                lo_hz=80.0,
                hi_hz=7600.0,
                bins=64,
                fft_size=2048,
                overlap=0.75,
                pad_end=False):
  mel = calc_mel(audio, lo_hz, hi_hz, bins, fft_size, overlap, pad_end)
  return safe_log(mel)


@gin.configurable
def calc_mfcc(audio,
              lo_hz=20.0,
              hi_hz=8000.0,
              fft_size=1024,
              mel_bins=128,
              mfcc_bins=13,
              overlap=0.75,
              pad_end=False):
  """Calculate Mel-frequency Cepstral Coefficients."""
  logmel = calc_logmel(
      audio,
      lo_hz=lo_hz,
      hi_hz=hi_hz,
      bins=mel_bins,
      fft_size=fft_size,
      overlap=overlap,
      pad_end=pad_end)
  mfccs = tf.signal.mfccs_from_log_mel_spectrograms(logmel)
  return mfccs[..., :mfcc_bins]


def get_loudness(audio, n_fft=2048, top_db=80.0, pmin=1e-20):
  """Perceptual loudness, following librosa implementation."""
  log10 = lambda x: tf.log(x) / tf.log(10.0)
  spectra = stft(audio, frame_size=n_fft, overlap=0.75)
  power = tf.abs(spectra)**2.0

  power_db = 10.0 * log10(tf.maximum(pmin, power))
  power_db = tf.maximum(power_db, tf.reduce_max(power_db) - top_db)

  fft_frequencies = librosa.fft_frequencies(n_fft=n_fft)
  a_weighting = librosa.A_weighting(fft_frequencies)

  loudness = power_db + a_weighting[tf.newaxis, tf.newaxis, :]
  loudness = tf.reduce_mean(loudness, axis=2)
  return loudness


def diff(x, axis=-1):
  """Take the finite difference of a tensor along an axis.

  Args:
    x: Input tensor of any dimension.
    axis: Axis on which to take the finite difference.

  Returns:
    d: Tensor with size less than x by 1 along the difference dimension.

  Raises:
    ValueError: Axis out of range for tensor.
  """
  shape = x.get_shape()
  if axis >= len(shape):
    raise ValueError('Invalid axis index: %d for tensor with only %d axes.' %
                     (axis, len(shape)))

  begin_back = [0 for _ in range(len(shape))]
  begin_front = [0 for _ in range(len(shape))]
  begin_front[axis] = 1

  size = shape.as_list()
  size[axis] -= 1
  slice_front = tf.slice(x, begin_front, size)
  slice_back = tf.slice(x, begin_back, size)
  d = slice_front - slice_back
  return d
