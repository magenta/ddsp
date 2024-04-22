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

"""Library of FFT operations for loss functions and conditioning."""

import crepe
from ddsp import core
from ddsp.core import safe_log
from ddsp.core import tf_float32
import gin
import librosa
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

CREPE_SAMPLE_RATE = 16000
CREPE_FRAME_SIZE = 1024

F0_RANGE = 127.0  # MIDI.
DB_RANGE = core.DB_RANGE  # dB (80.0).


def stft(audio, frame_size=2048, overlap=0.75, pad_end=True):
  """Differentiable stft in tensorflow, computed in batch."""
  # Remove channel dim if present.
  audio = tf_float32(audio)
  if len(audio.shape) == 3:
    audio = tf.squeeze(audio, axis=-1)

  s = tf.signal.stft(
      signals=audio,
      frame_length=int(frame_size),
      frame_step=int(frame_size * (1.0 - overlap)),
      fft_length=None,  # Use enclosing power of 2.
      pad_end=pad_end)
  return s


def stft_np(audio, frame_size=2048, overlap=0.75, pad_end=True):
  """Non-differentiable stft using librosa, one example at a time."""
  assert frame_size * overlap % 2.0 == 0.0
  hop_size = int(frame_size * (1.0 - overlap))
  is_2d = (len(audio.shape) == 2)

  if pad_end:
    audio = pad(audio, frame_size, hop_size, 'same', axis=is_2d).numpy()

  def stft_fn(y):
    return librosa.stft(
        y=y, n_fft=int(frame_size), hop_length=hop_size, center=False).T

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


def get_framed_lengths(input_length, frame_size, hop_size, padding='center'):
  """Give a strided framing, such as tf.signal.frame, gives output lengths.

  Args:
    input_length: Original length along the dimension to be framed.
    frame_size: Size of frames for striding.
    hop_size: Striding, space between frames.
    padding: Type of padding to apply, ['valid', 'same', 'center']. 'valid' is
      a no-op. 'same' applies padding to the end such that
      n_frames = n_t / hop_size. 'center' applies padding to both ends such that
      each frame timestamp is centered and n_frames = n_t / hop_size + 1.

  Returns:
    n_frames: Number of frames left after striding.
    padded_length: Length of the padded signal before striding.
  """
  # Use numpy since this function isn't used dynamically.
  def get_n_frames(length):
    return int(np.floor((length - frame_size) // hop_size)) + 1

  if padding == 'valid':
    padded_length = input_length
    n_frames = get_n_frames(input_length)

  elif padding == 'center':
    padded_length = input_length + frame_size
    n_frames = get_n_frames(padded_length)

  elif padding == 'same':
    n_frames = int(np.ceil(input_length / hop_size))
    padded_length = (n_frames - 1) * hop_size + frame_size

  return n_frames, padded_length


def pad(x, frame_size, hop_size, padding='center',
        axis=1, mode='CONSTANT', constant_values=0):
  """Pad a tensor for strided framing such as tf.signal.frame.

  Args:
    x: Tensor to pad, any shape.
    frame_size: Size of frames for striding.
    hop_size: Striding, space between frames.
    padding: Type of padding to apply, ['valid', 'same', 'center']. 'valid' is
      a no-op. 'same' applies padding to the end such that
      n_frames = n_t / hop_size. 'center' applies padding to both ends such that
      each frame timestamp is centered and n_frames = n_t / hop_size + 1.
    axis: Axis along which to pad `x`.
    mode: Padding mode for tf.pad(). One of "CONSTANT", "REFLECT", or
      "SYMMETRIC" (case-insensitive).
    constant_values: Passthrough kwarg for tf.pad().

  Returns:
    A padded version of `x` along axis. Output sizes can be computed separately
      with strided_lengths.
  """
  x = tf_float32(x)

  if padding == 'valid':
    return x

  if hop_size > frame_size:
    raise ValueError(f'During padding, frame_size ({frame_size})'
                     f' must be greater than hop_size ({hop_size}).')

  if len(x.shape) <= 1:
    axis = 0

  n_t = x.shape[axis]
  _, n_t_padded = get_framed_lengths(n_t, frame_size, hop_size, padding)
  pads = [[0, 0] for _ in range(len(x.shape))]

  if padding == 'same':
    pad_amount = int(n_t_padded - n_t)
    pads[axis] = [0, pad_amount]

  elif padding == 'center':
    pad_amount = int(frame_size // 2)  # Symmetric even padding like librosa.
    pads[axis] = [pad_amount, pad_amount]

  else:
    raise ValueError('`padding` must be one of [\'center\', \'same\''
                     f'\'valid\'], received ({padding}).')

  return tf.pad(x, pads, mode=mode, constant_values=constant_values)


def compute_rms_energy(audio,
                       sample_rate=16000,
                       frame_rate=250,
                       frame_size=512,
                       padding='center'):
  """Compute root mean squared energy of audio."""
  audio = tf_float32(audio)
  hop_size = sample_rate // frame_rate
  audio = pad(audio, frame_size, hop_size, padding=padding)
  audio_frames = tf.signal.frame(audio, frame_size, hop_size, pad_end=False)
  rms_energy = tf.reduce_mean(audio_frames**2.0, axis=-1)**0.5
  return rms_energy


def compute_power(audio,
                  sample_rate=16000,
                  frame_rate=250,
                  frame_size=512,
                  ref_db=0.0,
                  range_db=DB_RANGE,
                  padding='center'):
  """Compute power of audio in dB."""
  rms_energy = compute_rms_energy(
      audio, sample_rate, frame_rate, frame_size, padding=padding)
  power_db = core.amplitude_to_db(
      rms_energy, ref_db=ref_db, range_db=range_db, use_tf=True)
  return power_db


@gin.register
def compute_loudness(audio,
                     sample_rate=16000,
                     frame_rate=250,
                     n_fft=512,
                     range_db=DB_RANGE,
                     ref_db=0.0,
                     use_tf=True,
                     padding='center'):
  """Perceptual loudness (weighted power) in dB.

  Function is differentiable if use_tf=True.
  Args:
    audio: Numpy ndarray or tensor. Shape [batch_size, audio_length] or
      [audio_length,].
    sample_rate: Audio sample rate in Hz.
    frame_rate: Rate of loudness frames in Hz.
    n_fft: Fft window size.
    range_db: Sets the dynamic range of loudness in decibles. The minimum
      loudness (per a frequency bin) corresponds to -range_db.
    ref_db: Sets the reference maximum perceptual loudness as given by
      (A_weighting + 10 * log10(abs(stft(audio))**2.0). The old (<v2.0.0)
      default value corresponded to white noise with amplitude=1.0 and
      n_fft=2048. With v2.0.0 it was set to 0.0 to be more consistent with power
      calculations that have a natural scale for 0 dB being amplitude=1.0.
    use_tf: Make function differentiable by using tensorflow.
    padding: 'same', 'valid', or 'center'.

  Returns:
    Loudness in decibels. Shape [batch_size, n_frames] or [n_frames,].
  """
  # Pick tensorflow or numpy.
  lib = tf if use_tf else np
  reduce_mean = tf.reduce_mean if use_tf else np.mean
  stft_fn = stft if use_tf else stft_np

  # Make inputs tensors for tensorflow.
  frame_size = n_fft
  hop_size = sample_rate // frame_rate
  audio = pad(audio, frame_size, hop_size, padding=padding)
  audio = audio if use_tf else np.array(audio)

  # Temporarily a batch dimension for single examples.
  is_1d = (len(audio.shape) == 1)
  audio = audio[lib.newaxis, :] if is_1d else audio

  # Take STFT.
  overlap = 1 - hop_size / frame_size
  s = stft_fn(audio, frame_size=frame_size, overlap=overlap, pad_end=False)

  # Compute power.
  amplitude = lib.abs(s)
  power = amplitude**2

  # Perceptual weighting.
  frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
  a_weighting = librosa.A_weighting(frequencies)[lib.newaxis, lib.newaxis, :]

  # Perform weighting in linear scale, a_weighting given in decibels.
  weighting = 10**(a_weighting/10)
  power = power * weighting

  # Average over frequencies (weighted power per a bin).
  avg_power = reduce_mean(power, axis=-1)
  loudness = core.power_to_db(avg_power,
                              ref_db=ref_db,
                              range_db=range_db,
                              use_tf=use_tf)

  # Remove temporary batch dimension.
  loudness = loudness[0] if is_1d else loudness

  return loudness


@gin.register
def compute_f0(audio, frame_rate, viterbi=True, padding='center'):
  """Fundamental frequency (f0) estimate using CREPE.

  This function is non-differentiable and takes input as a numpy array.
  Args:
    audio: Numpy ndarray of single audio (16kHz) example. Shape [audio_length,].
    frame_rate: Rate of f0 frames in Hz.
    viterbi: Use Viterbi decoding to estimate f0.
    padding: Apply zero-padding for centered frames.
      'same', 'valid', or 'center'.

  Returns:
    f0_hz: Fundamental frequency in Hz. Shape [n_frames,].
    f0_confidence: Confidence in Hz estimate (scaled [0, 1]). Shape [n_frames,].
  """
  sample_rate = CREPE_SAMPLE_RATE
  crepe_step_size = 1000 / frame_rate  # milliseconds
  hop_size = sample_rate // frame_rate

  audio = pad(audio, CREPE_FRAME_SIZE, hop_size, padding)
  audio = np.asarray(audio)

  # Compute f0 with crepe.
  _, f0_hz, f0_confidence, _ = crepe.predict(
      audio,
      sr=sample_rate,
      viterbi=viterbi,
      step_size=crepe_step_size,
      center=False,
      verbose=0)

  # Postprocessing.
  f0_hz = f0_hz.astype(np.float32)
  f0_confidence = f0_confidence.astype(np.float32)
  f0_confidence = np.nan_to_num(f0_confidence)   # Set nans to 0 in confidence

  return f0_hz, f0_confidence


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
  on accelerators. For [full,large,small,tiny] crepe models, reads h5 models
  from installed pip package. Other saved models
  """

  def __init__(self,
               model_size_or_path,
               hop_size=160,
               **kwargs):
    super().__init__(**kwargs)
    self.hop_size = hop_size
    self.frame_size = 1024
    self.sample_rate = 16000
    # Load the crepe model.
    if model_size_or_path in ['full', 'large', 'small', 'tiny']:
      self.core_model = crepe.core.build_and_load_model(model_size_or_path)
    else:
      self.core_model = tf.keras.models.load_model(model_size_or_path)

    self.model_size_or_path = model_size_or_path

  @classmethod
  def activations_to_f0_and_confidence(cls, activations, centers=None):
    """Convert network outputs (activations) to f0 predictions."""
    cent_mapping = tf.cast(
        tf.linspace(0, 7180, 360) + 1997.3794084376191, tf.float32)

    # The confidence of voicing activity and the argmax bin.
    confidence = tf.reduce_max(activations, axis=-1, keepdims=True)
    if centers is None:
      centers = tf.math.argmax(activations, axis=-1)
    centers = tf.cast(centers, tf.int32)

    # Slice the local neighborhood around the argmax bin.
    start = centers - 4
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

  def predict_f0_and_confidence(self, audio, viterbi=False, padding='center'):
    audio = audio[None, :] if len(audio.shape) == 1 else audio
    batch_size = audio.shape[0]

    audio = pad(audio, self.frame_size, self.hop_size, padding=padding)

    frames = self.batch_frames(audio)
    frames = self.normalize_frames(frames)
    acts = self.core_model(frames, training=False)

    if viterbi:
      acts_viterbi = tf.reshape(acts, [batch_size, -1, 360])
      centers = self.viterbi_decode(acts_viterbi)
      centers = tf.reshape(centers, [-1])
    else:
      centers = None

    f0_hz, confidence = self.activations_to_f0_and_confidence(acts, centers)
    f0_hz = tf.reshape(f0_hz, [batch_size, -1])
    confidence = tf.reshape(confidence, [batch_size, -1])
    return f0_hz, confidence

  def create_hmm(self, num_steps):
    """Same as the original CREPE viterbi decdoding, but in TF."""
    # Initial distribution is uniform.
    initial_distribution = tfp.distributions.Categorical(
        probs=tf.ones([360]) / 360)

    # Transition probabilities inducing continuous pitch.
    bins = tf.range(360, dtype=tf.float32)
    xx, yy = tf.meshgrid(bins, bins)
    min_transition = 1e-5  # For training stabiity.
    transition = tf.maximum(12 - abs(xx - yy), min_transition)
    transition = transition / tf.reduce_sum(transition, axis=1)[:, None]
    transition = tf.cast(transition, tf.float32)
    transition_distribution = tfp.distributions.Categorical(
        probs=transition)

    # Emission probability = fixed probability for self, evenly distribute the
    # others.
    self_emission = 0.1
    emission = (
        tf.eye(360) * self_emission + tf.ones(shape=(360, 360)) *
        ((1 - self_emission) / 360.)
    )
    emission = tf.cast(emission, tf.float32)[None, ...]
    observation_distribution = tfp.distributions.Multinomial(
        total_count=1, probs=emission)

    return tfp.distributions.HiddenMarkovModel(
        initial_distribution=initial_distribution,
        transition_distribution=transition_distribution,
        observation_distribution=observation_distribution,
        num_steps=num_steps,
    )

  def viterbi_decode(self, acts):
    """Adapted from original CREPE viterbi decdoding, but in TF."""
    num_steps = acts.shape[1]
    hmm = self.create_hmm(num_steps)
    centers = hmm.posterior_mode(acts)
    return centers


