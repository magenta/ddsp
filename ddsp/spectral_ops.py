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
"""Library of FFT operations for loss functions and conditioning."""

import crepe
from ddsp.core import tf_float32
import gin
import librosa
import numpy as np
import tensorflow.compat.v2 as tf

_CREPE_SAMPLE_RATE = 16000
_CREPE_FRAME_SIZE = 1024

F0_RANGE = 127.0  # MIDI
LD_RANGE = 120.0  # dB


def safe_log(x, eps=1e-5):
  return tf.math.log(x + eps)


def stft(audio, frame_size=2048, overlap=0.75, pad_end=True):
  """Differentiable stft in tensorflow, computed in batch."""
  audio = tf_float32(audio)
  assert frame_size * overlap % 2.0 == 0.0
  s = tf.signal.stft(
      signals=audio,
      frame_length=int(frame_size),
      frame_step=int(frame_size * (1.0 - overlap)),
      fft_length=int(frame_size),
      pad_end=pad_end)
  return s


def stft_np(audio, frame_size=2048, overlap=0.75, pad_end=True):
  """Non-differentiable stft using librosa, one example at a time."""
  assert frame_size * overlap % 2.0 == 0.0
  hop_size = int(frame_size * (1.0 - overlap))
  is_2d = (len(audio.shape) == 2)

  if pad_end:
    n_samples_initial = int(audio.shape[-1])
    n_frames = int(np.ceil(n_samples_initial / hop_size))
    n_samples_final = (n_frames - 1) * hop_size + frame_size
    pad = n_samples_final - n_samples_initial
    padding = ((0, 0), (0, pad)) if is_2d else ((0, pad),)
    audio = np.pad(audio, padding, 'constant')

  def stft_fn(y):
    return librosa.stft(y=y,
                        n_fft=int(frame_size),
                        hop_length=hop_size,
                        center=False).T

  s = np.stack([stft_fn(a) for a in audio]) if is_2d else stft_fn(audio)
  return s


@gin.register
def compute_mag(audio, size=2048, overlap=0.75, pad_end=True):
  mag = tf.abs(stft(audio, frame_size=size, overlap=overlap, pad_end=pad_end))
  return tf_float32(mag)


@gin.register
def compute_mel(audio,
                lo_hz=0.0,
                hi_hz=8000.0,
                bins=64,
                fft_size=2048,
                overlap=0.75,
                pad_end=True):
  """Calculate Mel Spectrogram."""
  mag = compute_mag(audio, fft_size, overlap, pad_end)
  num_spectrogram_bins = int(mag.shape[-1])
  linear_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
      bins, num_spectrogram_bins, 16000, lo_hz, hi_hz)
  mel = tf.tensordot(mag, linear_to_mel_matrix, 1)
  mel.set_shape(mag.shape[:-1].concatenate(linear_to_mel_matrix.shape[-1:]))
  return mel


@gin.register
def compute_logmag(audio, size=2048):
  return safe_log(compute_mag(audio, size))


@gin.register
def compute_logmel(audio,
                   lo_hz=80.0,
                   hi_hz=7600.0,
                   bins=64,
                   fft_size=2048,
                   overlap=0.75,
                   pad_end=True):
  mel = compute_mel(audio, lo_hz, hi_hz, bins, fft_size, overlap, pad_end)
  return safe_log(mel)


@gin.register
def compute_mfcc(audio,
                 lo_hz=20.0,
                 hi_hz=8000.0,
                 fft_size=1024,
                 mel_bins=128,
                 mfcc_bins=13,
                 overlap=0.75,
                 pad_end=True):
  """Calculate Mel-frequency Cepstral Coefficients."""
  logmel = compute_logmel(
      audio,
      lo_hz=lo_hz,
      hi_hz=hi_hz,
      bins=mel_bins,
      fft_size=fft_size,
      overlap=overlap,
      pad_end=pad_end)
  mfccs = tf.signal.mfccs_from_log_mel_spectrograms(logmel)
  return mfccs[..., :mfcc_bins]


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
  shape = x.shape.as_list()
  if axis >= len(shape):
    raise ValueError('Invalid axis index: %d for tensor with only %d axes.' %
                     (axis, len(shape)))

  begin_back = [0 for _ in range(len(shape))]
  begin_front = [0 for _ in range(len(shape))]
  begin_front[axis] = 1

  shape[axis] -= 1
  slice_front = tf.slice(x, begin_front, shape)
  slice_back = tf.slice(x, begin_back, shape)
  d = slice_front - slice_back
  return d


@gin.register
def compute_loudness(audio,
                     sample_rate=16000,
                     frame_rate=250,
                     n_fft=2048,
                     range_db=LD_RANGE,
                     ref_db=20.7,
                     use_tf=False):
  """Perceptual loudness in dB, relative to white noise, amplitude=1.

  Function is differentiable if use_tf=True.
  Args:
    audio: Numpy ndarray or tensor. Shape [batch_size, audio_length] or
      [batch_size,].
    sample_rate: Audio sample rate in Hz.
    frame_rate: Rate of loudness frames in Hz.
    n_fft: Fft window size.
    range_db: Sets the dynamic range of loudness in decibles. The minimum
      loudness (per a frequency bin) corresponds to -range_db.
    ref_db: Sets the reference maximum perceptual loudness as given by
      (A_weighting + 10 * log10(abs(stft(audio))**2.0). The default value
      corresponds to white noise with amplitude=1.0 and n_fft=2048. There is a
      slight dependence on fft_size due to different granularity of perceptual
      weighting.
    use_tf: Make function differentiable by using librosa.

  Returns:
    Loudness in decibels. Shape [batch_size, n_frames] or [n_frames,].
  """
  # Pick tensorflow or numpy.
  lib = tf if use_tf else np

  # Make inputs tensors for tensorflow.
  audio = tf_float32(audio) if use_tf else audio

  # Temporarily a batch dimension for single examples.
  is_1d = (len(audio.shape) == 1)
  audio = audio[lib.newaxis, :] if is_1d else audio

  # Take STFT.
  hop_size = sample_rate // frame_rate
  overlap = 1 - hop_size / n_fft
  stft_fn = stft if use_tf else stft_np
  s = stft_fn(audio, frame_size=n_fft, overlap=overlap, pad_end=True)

  # Compute power
  amplitude = lib.abs(s)
  log10 = (lambda x: tf.math.log(x) / tf.math.log(10.0)) if use_tf else np.log10
  amin = 1e-20  # Avoid log(0) instabilities.
  power_db = log10(lib.maximum(amin, amplitude))
  power_db *= 20.0

  # Perceptual weighting.
  frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
  a_weighting = librosa.A_weighting(frequencies)[lib.newaxis, lib.newaxis, :]
  loudness = power_db + a_weighting

  # Set dynamic range.
  loudness -= ref_db
  loudness = lib.maximum(loudness, -range_db)
  mean = tf.reduce_mean if use_tf else np.mean

  # Average over frequency bins.
  loudness = mean(loudness, axis=-1)

  # Remove temporary batch dimension.
  loudness = loudness[0] if is_1d else loudness
  return loudness


@gin.register
def compute_f0(audio, sample_rate, frame_rate, viterbi=True):
  """Fundamental frequency (f0) estimate using CREPE.

  This function is non-differentiable and takes input as a numpy array.
  Args:
    audio: Numpy ndarray of single audio example. Shape [audio_length,].
    sample_rate: Sample rate in Hz.
    frame_rate: Rate of f0 frames in Hz.
    viterbi: Use Viterbi decoding to estimate f0.

  Returns:
    f0_hz: Fundamental frequency in Hz. Shape [n_frames,].
    f0_confidence: Confidence in Hz estimate (scaled [0, 1]). Shape [n_frames,].
  """

  # Pad end so that `n_frames = n_sec * frame_rate`.
  # We compute this using the CREPE model's sample rate and then convert back to
  # our sample rate.
  n_secs = len(audio) / sample_rate
  n_samples = n_secs * _CREPE_SAMPLE_RATE
  hop_size = sample_rate / frame_rate
  frame_size = _CREPE_FRAME_SIZE
  n_frames = np.ceil(n_secs * frame_rate)
  n_samples_padded = (n_frames - 1) * hop_size + frame_size
  n_padding = (n_samples_padded - n_samples) * sample_rate / _CREPE_SAMPLE_RATE
  assert n_padding % 1 == 0
  audio = np.pad(audio, (0, int(n_padding)), mode='constant')
  crepe_step_size = 1000 / frame_rate  # milliseconds

  # Compute f0 with crepe.
  _, f0_hz, f0_confidence, _ = crepe.predict(
      audio,
      sr=sample_rate,
      viterbi=viterbi,
      step_size=crepe_step_size,
      center=False,
      verbose=0)

  # Postprocessing.
  f0_confidence = np.nan_to_num(f0_confidence)   # Set nans to 0 in confidence.
  f0_hz = f0_hz.astype(np.float32)
  f0_confidence = f0_confidence.astype(np.float32)
  return f0_hz, f0_confidence


def reset_crepe():
  """Reset the global state of CREPE to force model re-building."""
  for k in crepe.core.models:
    crepe.core.models[k] = None
