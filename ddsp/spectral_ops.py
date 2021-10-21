# Copyright 2021 The DDSP Authors.
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
from ddsp.core import safe_log
from ddsp.core import tf_float32
import gin
import librosa
import numpy as np
import tensorflow.compat.v2 as tf

CREPE_SAMPLE_RATE = 16000
_CREPE_FRAME_SIZE = 1024

F0_RANGE = 127.0  # MIDI
LD_RANGE = 120.0  # dB


def stft(audio, frame_size=2048, overlap=0.75, pad_end=True):
  """Differentiable stft in tensorflow, computed in batch."""
  assert frame_size * overlap % 2.0 == 0.0

  # Remove channel dim if present.
  audio = tf_float32(audio)
  if len(audio.shape) == 3:
    audio = tf.squeeze(audio, axis=-1)

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
                pad_end=True,
                sample_rate=16000):
  """Calculate Mel Spectrogram."""
  mag = compute_mag(audio, fft_size, overlap, pad_end)
  num_spectrogram_bins = int(mag.shape[-1])
  linear_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
      bins, num_spectrogram_bins, sample_rate, lo_hz, hi_hz)
  mel = tf.tensordot(mag, linear_to_mel_matrix, 1)
  mel.set_shape(mag.shape[:-1].concatenate(linear_to_mel_matrix.shape[-1:]))
  return mel


@gin.register
def compute_logmag(audio, size=2048, overlap=0.75, pad_end=True):
  return safe_log(compute_mag(audio, size, overlap, pad_end))


@gin.register
def compute_logmel(audio,
                   lo_hz=80.0,
                   hi_hz=7600.0,
                   bins=64,
                   fft_size=2048,
                   overlap=0.75,
                   pad_end=True,
                   sample_rate=16000):
  """Logarithmic amplitude of mel-scaled spectrogram."""
  mel = compute_mel(audio, lo_hz, hi_hz, bins,
                    fft_size, overlap, pad_end, sample_rate)
  return safe_log(mel)


@gin.register
def compute_mfcc(audio,
                 lo_hz=20.0,
                 hi_hz=8000.0,
                 fft_size=1024,
                 mel_bins=128,
                 mfcc_bins=13,
                 overlap=0.75,
                 pad_end=True,
                 sample_rate=16000):
  """Calculate Mel-frequency Cepstral Coefficients."""
  logmel = compute_logmel(
      audio,
      lo_hz=lo_hz,
      hi_hz=hi_hz,
      bins=mel_bins,
      fft_size=fft_size,
      overlap=overlap,
      pad_end=pad_end,
      sample_rate=sample_rate)
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
  ndim = len(shape)
  if axis >= ndim:
    raise ValueError('Invalid axis index: %d for tensor with only %d axes.' %
                     (axis, ndim))

  begin_back = [0 for _ in range(ndim)]
  begin_front = [0 for _ in range(ndim)]
  begin_front[axis] = 1

  shape[axis] -= 1
  slice_front = tf.slice(x, begin_front, shape)
  slice_back = tf.slice(x, begin_back, shape)
  d = slice_front - slice_back
  return d


def amplitude_to_db(amplitude, use_tf=False):
  """Converts amplitude to decibels."""
  lib = tf if use_tf else np
  log10 = (lambda x: tf.math.log(x) / tf.math.log(10.0)) if use_tf else np.log10
  amin = 1e-20  # Avoid log(0) instabilities.
  db = log10(lib.maximum(amin, amplitude))
  db *= 20.0
  return db


def db_to_amplitude(db):
  """Converts decibels to amplitude."""
  return 10.0**(db / 20.0)


@gin.register
def compute_loudness(audio,
                     sample_rate=16000,
                     frame_rate=250,
                     n_fft=2048,
                     range_db=LD_RANGE,
                     ref_db=20.7,
                     use_tf=False,
                     pad_end=True):
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
    use_tf: Make function differentiable by using tensorflow.
    pad_end: Add zero padding at end of audio (like `same` convolution).

  Returns:
    Loudness in decibels. Shape [batch_size, n_frames] or [n_frames,].
  """
  if sample_rate % frame_rate != 0:
    raise ValueError(
        'frame_rate: {} must evenly divide sample_rate: {}.'
        'For default frame_rate: 250Hz, suggested sample_rate: 16kHz or 48kHz'
        .format(frame_rate, sample_rate))

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
  s = stft_fn(audio, frame_size=n_fft, overlap=overlap, pad_end=pad_end)

  # Compute power.
  amplitude = lib.abs(s)
  power_db = amplitude_to_db(amplitude, use_tf=use_tf)

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

  # Compute expected length of loudness vector
  n_secs = audio.shape[-1] / float(
      sample_rate)  # `n_secs` can have milliseconds
  expected_len = int(n_secs * frame_rate)

  # Pad with `-range_db` noise floor or trim vector
  loudness = pad_or_trim_to_expected_length(
      loudness, expected_len, -range_db, use_tf=use_tf)
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

  n_secs = len(audio) / float(sample_rate)  # `n_secs` can have milliseconds
  crepe_step_size = 1000 / frame_rate  # milliseconds
  expected_len = int(n_secs * frame_rate)
  audio = np.asarray(audio)

  # Compute f0 with crepe.
  _, f0_hz, f0_confidence, _ = crepe.predict(
      audio,
      sr=sample_rate,
      viterbi=viterbi,
      step_size=crepe_step_size,
      center=False,
      verbose=0)

  # Postprocessing on f0_hz
  f0_hz = pad_or_trim_to_expected_length(f0_hz, expected_len, 0)  # pad with 0
  f0_hz = f0_hz.astype(np.float32)

  # Postprocessing on f0_confidence
  f0_confidence = pad_or_trim_to_expected_length(f0_confidence, expected_len, 1)
  f0_confidence = np.nan_to_num(f0_confidence)   # Set nans to 0 in confidence
  f0_confidence = f0_confidence.astype(np.float32)
  return f0_hz, f0_confidence


def compute_rms_energy(audio,
                       sample_rate=16000,
                       frame_rate=250,
                       frame_size=2048,
                       pad_end=True):
  """Compute root mean squared energy of audio."""
  audio = tf_float32(audio)
  hop_size = sample_rate // frame_rate
  audio_frames = tf.signal.frame(audio, frame_size, hop_size, pad_end=pad_end)
  rms_energy = tf.reduce_mean(audio_frames**2.0, axis=-1)**0.5
  if pad_end:
    n_samples = audio.shape[0] if len(audio.shape) == 1 else audio.shape[1]
    n_secs = n_samples / float(sample_rate)  # `n_secs` can have milliseconds
    expected_len = int(n_secs * frame_rate)
    return pad_or_trim_to_expected_length(rms_energy, expected_len, use_tf=True)
  else:
    return rms_energy


def compute_power(audio,
                  sample_rate=16000,
                  frame_rate=250,
                  frame_size=1024,
                  range_db=LD_RANGE,
                  ref_db=20.7,
                  pad_end=True):
  """Compute power of audio in dB."""
  # TODO(hanoih@): enable `use_tf` to be True or False like `compute_loudness`
  rms_energy = compute_rms_energy(
      audio, sample_rate, frame_rate, frame_size, pad_end)
  power_db = amplitude_to_db(rms_energy**2, use_tf=True)
  # Set dynamic range.
  power_db -= ref_db
  power_db = tf.maximum(power_db, -range_db)
  return power_db


def pad_or_trim_to_expected_length(vector,
                                   expected_len,
                                   pad_value=0,
                                   len_tolerance=20,
                                   use_tf=False):
  """Make vector equal to the expected length.

  Feature extraction functions like `compute_loudness()` or `compute_f0` produce
  feature vectors that vary in length depending on factors such as `sample_rate`
  or `hop_size`. This function corrects vectors to the expected length, warning
  the user if the difference between the vector and expected length was
  unusually high to begin with.

  Args:
    vector: Numpy 1D ndarray. Shape [vector_length,]
    expected_len: Expected length of vector.
    pad_value: Value to pad at end of vector.
    len_tolerance: Tolerance of difference between original and desired vector
      length.
    use_tf: Make function differentiable by using tensorflow.

  Returns:
    vector: Vector with corrected length.

  Raises:
    ValueError: if `len(vector)` is different from `expected_len` beyond
    `len_tolerance` to begin with.
  """
  expected_len = int(expected_len)
  vector_len = int(vector.shape[-1])

  if abs(vector_len - expected_len) > len_tolerance:
    # Ensure vector was close to expected length to begin with
    raise ValueError('Vector length: {} differs from expected length: {} '
                     'beyond tolerance of : {}'.format(vector_len,
                                                       expected_len,
                                                       len_tolerance))
  # Pick tensorflow or numpy.
  lib = tf if use_tf else np

  is_1d = (len(vector.shape) == 1)
  vector = vector[lib.newaxis, :] if is_1d else vector

  # Pad missing samples
  if vector_len < expected_len:
    n_padding = expected_len - vector_len
    vector = lib.pad(
        vector, ((0, 0), (0, n_padding)),
        mode='constant',
        constant_values=pad_value)
  # Trim samples
  elif vector_len > expected_len:
    vector = vector[..., :expected_len]

  # Remove temporary batch dimension.
  vector = vector[0] if is_1d else vector
  return vector


def reset_crepe():
  """Reset the global state of CREPE to force model re-building."""
  for k in crepe.core.models:
    crepe.core.models[k] = None


class PretrainedCREPE(tf.keras.Model):
  """A wrapper around a pretrained CREPE model, for pitch prediction.

  Enables predicting pitch and confidence entirely in TF for running in batch
  on accelerators. Constructor requires path to a SavedModel of the base CREPE
  models. Available on GCS at gs://crepe-models/saved_models/[full,large,small].
  """

  def __init__(self,
               saved_model_path,
               hop_size=160,
               **kwargs):
    super().__init__(**kwargs)
    self.hop_size = hop_size
    self.frame_size = 1024
    self.sample_rate = 16000
    # Load the crepe model.
    self.saved_model_path = saved_model_path
    self.core_model = tf.keras.models.load_model(self.saved_model_path)

  @classmethod
  def activations_to_f0_and_confidence(cls, activations):
    """Convert network outputs (activations) to f0 predictions."""
    cent_mapping = tf.cast(
        tf.linspace(0, 7180, 360) + 1997.3794084376191, tf.float32)

    # The confidence of voicing activity and the argmax bin.
    confidence = tf.reduce_max(activations, axis=-1, keepdims=True)
    center = tf.cast(tf.math.argmax(activations, axis=-1), tf.int32)

    # Slice the local neighborhood around the argmax bin.
    start = center - 4
    idx_list = tf.range(0, 10)
    idx_list = start[:, None] + idx_list[None, :]

    # Bound to [0, 359].
    idx_list = tf.where(idx_list > 0, idx_list, 0)
    idx_list = tf.where(idx_list < 359, idx_list, 359)

    # Gather and weight activations.
    weights = tf.gather(activations, idx_list, batch_dims=1)
    cents = tf.gather(cent_mapping, idx_list, batch_dims=0)
    f0_cent = tf.reduce_sum(weights * cents, axis=-1) / tf.reduce_sum(
        weights, axis=-1)
    f0_hz = 10 * 2**(f0_cent / 1200.)

    return f0_hz, confidence

  def batch_frames(self, audio):
    """Chop audio into overlapping frames, and push to batch dimension."""
    if audio.shape[-1] == self.frame_size:
      return audio
    else:
      frames = tf.signal.frame(audio, self.frame_size, self.hop_size)
      frames = tf.reshape(frames, [-1, self.frame_size])
      return frames

  def normalize_frames(self, frames):
    """Normalize each frame -- this is expected by the model."""
    mu, var = tf.nn.moments(frames, axes=[-1])
    std = tf.where(tf.abs(var) > 0, tf.sqrt(var), 1e-8)
    frames -= mu[:, None]
    frames /= std[:, None]
    return frames

  def predict_f0_and_confidence(self, audio):
    audio = audio[None, :] if len(audio.shape) == 1 else audio
    batch_size = audio.shape[0]

    frames = self.batch_frames(audio)
    frames = self.normalize_frames(frames)
    acts = self.core_model(frames, training=False)
    f0_hz, confidence = self.activations_to_f0_and_confidence(acts)
    f0_hz = tf.reshape(f0_hz, [batch_size, -1])
    confidence = tf.reshape(confidence, [batch_size, -1])
    return f0_hz, confidence


