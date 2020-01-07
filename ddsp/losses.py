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
"""Library of loss functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from ddsp import pretrained_models
from ddsp import spectral_ops
from ddsp.core import tf_float32

import gin
import tensorflow.compat.v1 as tf


# ---------------------- Losses ------------------------------------------------
def mean_difference(target, value, loss_type='L1'):
  difference = target - value
  if loss_type == 'L1':
    return tf.reduce_mean(tf.abs(difference))
  elif loss_type == 'L2':
    return tf.reduce_mean(difference**2)
  elif loss_type == 'cosine':
    return tf.losses.cosine_distance(target, value, axis=-1)


class Loss(object):
  """Base class to implement any loss.

  Users should override compute_loss() to define the actual loss.
  Hyper-parameters of the loss will be passed through the constructor.
  """

  def __init__(self, name):
    self.name = name
    self.pretrained_model = None

  def __call__(self, *args, **kwargs):
    """Alias to compute_loss."""
    return self.compute_loss(*args, **kwargs)

  def compute_loss(self, audio, target_audio):
    """Subclasses must implement compute_loss().

    Args:
      audio: 2D tensor of shape [batch, time].
      target_audio: 2D tensor of shape [batch, time].

    Returns:
      A scalar tensor of the loss.
    """
    raise NotImplementedError


@gin.register
class SpectralLoss(Loss):
  """Multi-scale spectrogram loss."""

  def __init__(self,
               fft_sizes=(2048, 1024, 512, 256, 128, 64),
               loss_type='L1',
               mag_weight=1.0,
               delta_time_weight=0.0,
               delta_delta_time_weight=0.0,
               delta_freq_weight=0.0,
               delta_delta_freq_weight=0.0,
               logmag_weight=0.0,
               loudness_weight=0.0,
               name='spectral_loss'):
    super(SpectralLoss, self).__init__(name=name)
    self.fft_sizes = fft_sizes
    self.loss_type = loss_type
    self.mag_weight = mag_weight
    self.delta_time_weight = delta_time_weight
    self.delta_delta_time_weight = delta_delta_time_weight
    self.delta_freq_weight = delta_freq_weight
    self.delta_delta_freq_weight = delta_delta_freq_weight
    self.logmag_weight = logmag_weight
    self.loudness_weight = loudness_weight

  def compute_loss(self, audio, target_audio):

    loss = 0.0
    loss_ops = []
    diff = spectral_ops.diff

    for size in self.fft_sizes:
      loss_op = functools.partial(spectral_ops.compute_mag, size=size)
      loss_ops.append(loss_op)

    # Compute loss for each fft size.
    for loss_op in loss_ops:
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

      if self.delta_delta_time_weight > 0:
        target = diff(diff(target_mag, axis=1), axis=1)
        value = diff(diff(value_mag, axis=1), axis=1)
        loss += self.delta_delta_time_weight * mean_difference(
            target, value, self.loss_type)

      if self.delta_freq_weight > 0:
        target = diff(target_mag, axis=2)
        value = diff(value_mag, axis=2)
        loss += self.delta_freq_weight * mean_difference(
            target, value, self.loss_type)

      if self.delta_delta_freq_weight > 0:
        target = diff(diff(target_mag, axis=2), axis=2)
        value = diff(diff(value_mag, axis=2), axis=2)
        loss += self.delta_delta_freq_weight * mean_difference(
            target, value, self.loss_type)

      # Add logmagnitude loss, reusing spectrogram.
      if self.logmag_weight > 0:
        target = spectral_ops.safe_log(target_mag)
        value = spectral_ops.safe_log(value_mag)
        loss += self.logmag_weight * mean_difference(target, value,
                                                     self.loss_type)

    if self.loudness_weight > 0:
      target = spectral_ops.compute_loudness(target_audio, n_fft=2048)
      value = spectral_ops.compute_loudness(audio, n_fft=2048)
      loss += self.loudness_weight * mean_difference(target, value,
                                                     self.loss_type)

    return loss


@gin.register
class EmbeddingLoss(Loss):
  """Embedding loss for a given pretrained model.

  Calculates the embedding loss given a pretrained model.

  You may also define a trivial pretrained model to apply any function that
  computes the embedding.
  """

  def __init__(self,
               weight=1.0,
               loss_type='L1',
               pretrained_model=None,
               name='embedding_loss'):
    super(EmbeddingLoss, self).__init__(name=name)
    self.weight = weight
    self.loss_type = loss_type
    self.pretrained_model = pretrained_model

  def compute_loss(self, audio, target_audio):
    audio, target_audio = tf_float32(audio), tf_float32(target_audio)
    target_emb = self.pretrained_model(target_audio)
    synth_emb = self.pretrained_model(audio)
    loss = self.weight * mean_difference(target_emb, synth_emb, self.loss_type)
    return loss


@gin.register
class PretrainedCREPEEmbeddingLoss(EmbeddingLoss):
  """Embedding loss of the CREPE model."""

  def __init__(self,
               weight=1.0,
               loss_type='L1',
               model_capacity='tiny',
               activation_layer='classifier',
               checkpoint='/path/to/crepe/model-tiny.ckpt',
               name='pretrained_crepe_embedding_loss'):
    # Scale each layer activation loss to comparable scales.
    scale = {
        'classifier': 6.8e-4,
        'conv6-maxpool': 2.5e4,
        'conv5-maxpool': 4.0e4,
        'conv4-maxpool': 4.9e3,
        'conv3-maxpool': 4.0e2,
        'conv2-maxpool': 3.0e1,
        'conv1-maxpool': 1.3e0,
        'conv6-BN': 1.8e4,
        'conv5-BN': 3.5e4,
        'conv4-BN': 3.9e3,
        'conv3-BN': 2.6e2,
        'conv2-BN': 2.1e1,
        'conv1-BN': 1.0e0,
    }[activation_layer]
    super(PretrainedCREPEEmbeddingLoss, self).__init__(
        weight=20 / scale * weight,
        loss_type=loss_type,
        name=name,
        pretrained_model=pretrained_models.PretrainedCREPE(
            model_capacity=model_capacity,
            activation_layer=activation_layer,
            checkpoint=checkpoint))


