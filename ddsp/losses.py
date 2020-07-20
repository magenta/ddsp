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
"""Library of loss functions."""

import functools

import crepe
from ddsp import spectral_ops
from ddsp.core import tf_float32

import gin
import tensorflow.compat.v2 as tf

tfkl = tf.keras.layers


# ---------------------- Losses ------------------------------------------------
def mean_difference(target, value, loss_type='L1', weights=None):
  """Common loss functions.

  Args:
    target: Target tensor.
    value: Value tensor.
    loss_type: One of 'L1', 'L2', or 'COSINE'.
    weights: A weighting mask for the per-element differences.

  Returns:
    The average loss.

  Raises:
    ValueError: If loss_type is not an allowed value.
  """
  difference = target - value
  weights = 1.0 if weights is None else weights
  loss_type = loss_type.upper()
  if loss_type == 'L1':
    return tf.reduce_mean(tf.abs(difference * weights))
  elif loss_type == 'L2':
    return tf.reduce_mean(difference**2 * weights)
  elif loss_type == 'COSINE':
    return tf.losses.cosine_distance(target, value, weights=weights, axis=-1)
  else:
    raise ValueError('Loss type ({}), must be '
                     '"L1", "L2", or "COSINE"'.format(loss_type))


@gin.register
class SpectralLoss(tfkl.Layer):
  """Multi-scale spectrogram loss.

  This loss is the bread-and-butter of comparing two audio signals. It offers
  a range of options to compare spectrograms, many of which are redunant, but
  emphasize different aspects of the signal. By far, the most common comparisons
  are magnitudes (mag_weight) and log magnitudes (logmag_weight).
  """

  def __init__(self,
               fft_sizes=(2048, 1024, 512, 256, 128, 64),
               loss_type='L1',
               mag_weight=1.0,
               delta_time_weight=0.0,
               delta_freq_weight=0.0,
               cumsum_freq_weight=0.0,
               logmag_weight=0.0,
               loudness_weight=0.0,
               name='spectral_loss'):
    """Constructor, set loss weights of various components.

    Args:
      fft_sizes: Compare spectrograms at each of this list of fft sizes. Each
        spectrogram has a time-frequency resolution trade-off based on fft size,
        so comparing multiple scales allows multiple resolutions.
      loss_type: One of 'L1', 'L2', or 'COSINE'.
      mag_weight: Weight to compare linear magnitudes of spectrograms. Core
        audio similarity loss. More sensitive to peak magnitudes than log
        magnitudes.
      delta_time_weight: Weight to compare the first finite difference of
        spectrograms in time. Emphasizes changes of magnitude in time, such as
        at transients.
      delta_freq_weight: Weight to compare the first finite difference of
        spectrograms in frequency. Emphasizes changes of magnitude in frequency,
        such as at the boundaries of a stack of harmonics.
      cumsum_freq_weight: Weight to compare the cumulative sum of spectrograms
        across frequency for each slice in time. Similar to a 1-D Wasserstein
        loss, this hopefully provides a non-vanishing gradient to push two
        non-overlapping sinusoids towards eachother.
      logmag_weight: Weight to compare log magnitudes of spectrograms. Core
        audio similarity loss. More sensitive to quiet magnitudes than linear
        magnitudes.
      loudness_weight: Weight to compare the overall perceptual loudness of two
        signals. Very high-level loss signal that is a subset of mag and
        logmag losses.
      name: Name of the module.
    """
    super().__init__(name=name)
    self.fft_sizes = fft_sizes
    self.loss_type = loss_type
    self.mag_weight = mag_weight
    self.delta_time_weight = delta_time_weight
    self.delta_freq_weight = delta_freq_weight
    self.cumsum_freq_weight = cumsum_freq_weight
    self.logmag_weight = logmag_weight
    self.loudness_weight = loudness_weight

    self.spectrogram_ops = []
    for size in self.fft_sizes:
      spectrogram_op = functools.partial(spectral_ops.compute_mag, size=size)
      self.spectrogram_ops.append(spectrogram_op)

  def call(self, target_audio, audio):

    loss = 0.0

    diff = spectral_ops.diff
    cumsum = tf.math.cumsum

    # Compute loss for each fft size.
    for loss_op in self.spectrogram_ops:
      target_mag = loss_op(target_audio)
      value_mag = loss_op(audio)

      # Add magnitude loss.
      if self.mag_weight > 0:
        loss += self.mag_weight * mean_difference(target_mag, value_mag,
                                                  self.loss_type)

      if self.delta_time_weight > 0:
        target = diff(target_mag, axis=1)
        value = diff(value_mag, axis=1)
        loss += self.delta_time_weight * mean_difference(
            target, value, self.loss_type)

      if self.delta_freq_weight > 0:
        target = diff(target_mag, axis=2)
        value = diff(value_mag, axis=2)
        loss += self.delta_freq_weight * mean_difference(
            target, value, self.loss_type)

      # TODO(kyriacos) normalize cumulative spectrogram
      if self.cumsum_freq_weight > 0:
        target = cumsum(target_mag, axis=2)
        value = cumsum(value_mag, axis=2)
        loss += self.cumsum_freq_weight * mean_difference(
            target, value, self.loss_type)

      # Add logmagnitude loss, reusing spectrogram.
      if self.logmag_weight > 0:
        target = spectral_ops.safe_log(target_mag)
        value = spectral_ops.safe_log(value_mag)
        loss += self.logmag_weight * mean_difference(target, value,
                                                     self.loss_type)

    if self.loudness_weight > 0:
      target = spectral_ops.compute_loudness(target_audio, n_fft=2048,
                                             use_tf=True)
      value = spectral_ops.compute_loudness(audio, n_fft=2048, use_tf=True)
      loss += self.loudness_weight * mean_difference(target, value,
                                                     self.loss_type)

    return loss


@gin.register
class EmbeddingLoss(tfkl.Layer):
  """Embedding loss for a given pretrained model.

  Using these "perceptual" loss functions will help encourage better matching
  of higher-order structure than just a spectral loss. In image models, these
  perceptual losses help overcome the tendancy of straightforward L1 and L2
  losses to lead to blurry images. For ddsp, a "blurry" image is often a
  filtered noise synthesizer or reverb smearing power density in the right areas
  of a spectrogram but without the appropriate harmonic structure. This
  perceptual loss encouages output from harmonic and sinusoidal models because
  the pretrained model compares the two audio signals with features that are
  trained to detect the harmonic structure of natural sounds.
  """

  def __init__(self,
               weight=1.0,
               loss_type='L1',
               pretrained_model=None,
               name='embedding_loss'):
    super().__init__(name=name)
    self.weight = weight
    self.loss_type = loss_type
    self.pretrained_model = pretrained_model

  def call(self, target_audio, audio):
    loss = 0.0
    if self.weight > 0.0:
      audio, target_audio = tf_float32(audio), tf_float32(target_audio)
      target_emb = self.pretrained_model(target_audio)
      synth_emb = self.pretrained_model(audio)
      loss = self.weight * mean_difference(
          target_emb, synth_emb, self.loss_type)
    return loss


@gin.register
class PretrainedCREPEEmbeddingLoss(EmbeddingLoss):
  """Embedding loss of a pretrained CREPE model."""

  def __init__(self,
               weight=1.0,
               loss_type='L1',
               model_capacity='tiny',
               activation_layer='classifier',
               name='pretrained_crepe_embedding_loss'):
    # Scale each layer activation loss to comparable scales.
    scale = {
        'conv1-BN': 1.3,
        'conv1-maxpool': 1.0,
        'conv2-BN': 1.4,
        'conv2-maxpool': 1.1,
        'conv3-BN': 1.9,
        'conv3-maxpool': 1.6,
        'conv4-BN': 1.5,
        'conv4-maxpool': 1.4,
        'conv5-BN': 1.9,
        'conv5-maxpool': 1.7,
        'conv6-BN': 30,
        'conv6-maxpool': 25,
        'classifier': 130,
    }[activation_layer]
    super().__init__(
        weight=20.0 * scale * weight,
        loss_type=loss_type,
        name=name,
        pretrained_model=PretrainedCREPE(model_capacity=model_capacity,
                                         activation_layer=activation_layer))


class PretrainedCREPE(tfkl.Layer):
  """Pretrained CREPE model with frozen weights."""

  def __init__(self,
               model_capacity='tiny',
               activation_layer='conv5-maxpool',
               name='pretrained_crepe',
               trainable=False):
    super(PretrainedCREPE, self).__init__(name=name, trainable=trainable)
    self._model_capacity = model_capacity
    self._activation_layer = activation_layer
    spectral_ops.reset_crepe()
    self._model = crepe.core.build_and_load_model(self._model_capacity)
    self.frame_length = 1024

  def build(self, unused_x_shape):
    self.layer_names = [l.name for l in self._model.layers]

    if self._activation_layer not in self.layer_names:
      raise ValueError(
          'activation layer {} not found, valid names are {}'.format(
              self._activation_layer, self.layer_names))

    self._activation_model = tf.keras.Model(
        inputs=self._model.input,
        outputs=self._model.get_layer(self._activation_layer).output)

    # Variables are not to be trained.
    self._model.trainable = self.trainable
    self._activation_model.trainable = self.trainable

  def frame_audio(self, audio, hop_length=1024, center=True):
    """Slice audio into frames for crepe."""
    # Pad so that frames are centered around their timestamps.
    # (i.e. first frame is zero centered).
    pad = int(self.frame_length / 2)
    audio = tf.pad(audio, ((0, 0), (pad, pad))) if center else audio
    frames = tf.signal.frame(audio,
                             frame_length=self.frame_length,
                             frame_step=hop_length)

    # Normalize each frame -- this is expected by the model.
    mean, var = tf.nn.moments(frames, [-1], keepdims=True)
    frames -= mean
    frames /= (var**0.5 + 1e-5)
    return frames

  def call(self, audio):
    """Returns the embeddings.

    Args:
      audio: tensors of shape [batch, length]. Length must be divisible by 1024.

    Returns:
      activations of shape [batch, depth]
    """
    frames = self.frame_audio(audio)
    batch_size = int(frames.shape[0])
    n_frames = int(frames.shape[1])
    # Get model predictions.
    frames = tf.reshape(frames, [-1, self.frame_length])
    outputs = self._activation_model(frames)
    outputs = tf.reshape(outputs, [batch_size, n_frames, -1])
    return outputs


