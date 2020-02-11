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
"""Library of pretrained models for use in perceptual loss functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import tensorflow.compat.v1 as tf

tfkl = tf.keras.layers


class UntrainedModel(object):
  """Base class that wrap any pretrained model."""

  def __init__(self, name=''):
    self._name = name

  @property
  def name(self):
    return self._name

  @property
  def variable_scope(self):
    return self._name

  def __call__(self, audio, training):
    with tf.variable_scope(self.variable_scope, reuse=tf.AUTO_REUSE):
      outputs = self.get_outputs(audio, training)
      return outputs

  def get_outputs(self, audio, training):
    """Returns the output of the model, usually an embedding."""
    raise NotImplementedError

  def trainable_variables(self):
    return tf.trainable_variables(scope=self.variable_scope)


class Crepe(tfkl.Layer):
  """Crepe model modified for 4-second (64000 sample) audio

    Paper: https://arxiv.org/pdf/1802.06182.pdf

    Adapted from: https://github.com/marl/crepe/blob/master/crepe/core.py
  """

  def __init__(self, model_capacity, activation_layer, name=None):
    super(Crepe, self).__init__(self, name=name)
    self._model_capacity = model_capacity
    self._capacity_multiplier = {
        'tiny': 4,
        'small': 8,
        'medium': 16,
        'large': 24,
        'full': 32
    }[model_capacity]
    self._activation_layer = activation_layer
    self._layers = self._build_layers()

  def _build_layers(self):
    """Returns layers in the CREPE model."""
    layers = []
    k = tf.keras.layers  # short-cut for readability

    layer_ids = [1, 2, 3, 4, 5, 6]
    filters = [n * self._capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
    widths = [512, 64, 64, 64, 64, 64]
    # note that unlike original crepe, there's no stride of (4, 1) at layer=1
    # because we need 250 f0 prediction per second
    strides = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

    for l, f, w, s in zip(layer_ids, filters, widths, strides):
      layers.append(
          k.Conv2D(
              f, (w, 1),
              strides=s,
              padding='same',
              activation='relu',
              name='conv%d' % l))

      layers.append(k.BatchNormalization(name='conv%d-BN' % l))
      layers.append(
          k.MaxPool2D(
              pool_size=(2, 1),
              strides=None,
              padding='valid',
              name='conv%d-maxpool' % l))
      layers.append(k.Dropout(0.25, name='conv%d-dropout' % l))

    # layers.append(k.Permute((2, 1, 3), name='transpose'))
    # layers.append(k.Flatten(name='flatten'))
    # layers.append(k.Dense(360, activation='sigmoid', name='classifier'))
    layers.append(k.Conv2D(360, (4, 1),
                           padding='same',
                           activation='sigmoid',
                           name='dense_alternative'))
    layers.append(k.Reshape((1000, 360), name='classifier'))

    return layers

  def call(self, audio, training=False):
    # returns a dict of tensors, from layer name to layer activations.
    assert audio.shape[1] == 64000, audio.shape

    y = audio
    for _ in range(2):
      y = tf.expand_dims(y, axis=-1)  # [batch, length, 1, 1]

    activation_dict = {}

    for layer in self._layers:
      if 'BN' in layer.name or 'dropout' in layer.name:
        y = layer(y, training=training)
      else:
        y = layer(y)
      activation_dict[layer.name] = y
    return activation_dict  # final shape: (batch, 1000, 360)


class TrainableCREPE(UntrainedModel):
  """CREPE model."""

  def __init__(self,
               model_capacity='tiny',
               activation_layer='classifier',
               name='crepe'):
    super(TrainableCREPE, self).__init__(name=name)
    self._model_capacity = model_capacity
    self._activation_layer = activation_layer
    self._model = None

  def _build_model(self):
    """Builds the CREPE model."""
    self._model = Crepe(
        self._model_capacity, self._activation_layer, name=self._name)

  def __call__(self, audio, training):
    return self.get_outputs(audio, training)

  def get_outputs(self, audio, training):
    """Returns the embeddings.

    Args:
      audio: tensors of shape [batch, length=64000].

    Returns:
      activations of shape [batch, depth]
    """
    if self._model is None:
      self._build_model()
    batch_size = tf.shape(audio)[0]

    # TODO(gcj): relax this constraint by modifying the model to generate
    # outputs at every time point.
    activation_dict = self._model(audio, training)
    if self._activation_layer not in activation_dict:
      raise ValueError(
          'activation layer {} not found, valid keys are {}'.format(
              self._activation_layer, sorted(activation_dict.keys())))
    outputs = activation_dict.get(self._activation_layer)
    # outputs = tf.reshape(outputs, [batch_size, -1])

    # post-processing to compute cent and hz should happen outside hereafter.
    return outputs


"""
Layer (type)                 Output Shape              Param #   
=================================================================
input (InputLayer)           [(None, 64000)]           0         
_________________________________________________________________
input-reshape (Reshape)      (None, 64000, 1, 1)       0         
_________________________________________________________________
conv1 (Conv2D)               (None, 64000, 1, 128)     65664     
_________________________________________________________________
conv1-BN (BatchNormalization (None, 64000, 1, 128)     512       
_________________________________________________________________
conv1-maxpool (MaxPooling2D) (None, 32000, 1, 128)     0         
_________________________________________________________________
conv1-dropout (Dropout)      (None, 32000, 1, 128)     0         
_________________________________________________________________
conv2 (Conv2D)               (None, 32000, 1, 16)      131088    
_________________________________________________________________
conv2-BN (BatchNormalization (None, 32000, 1, 16)      64        
_________________________________________________________________
conv2-maxpool (MaxPooling2D) (None, 16000, 1, 16)      0         
_________________________________________________________________
conv2-dropout (Dropout)      (None, 16000, 1, 16)      0         
_________________________________________________________________
conv3 (Conv2D)               (None, 16000, 1, 16)      16400     
_________________________________________________________________
conv3-BN (BatchNormalization (None, 16000, 1, 16)      64        
_________________________________________________________________
conv3-maxpool (MaxPooling2D) (None, 8000, 1, 16)       0         
_________________________________________________________________
conv3-dropout (Dropout)      (None, 8000, 1, 16)       0         
_________________________________________________________________
conv4 (Conv2D)               (None, 8000, 1, 16)       16400     
_________________________________________________________________
conv4-BN (BatchNormalization (None, 8000, 1, 16)       64        
_________________________________________________________________
conv4-maxpool (MaxPooling2D) (None, 4000, 1, 16)       0         
_________________________________________________________________
conv4-dropout (Dropout)      (None, 4000, 1, 16)       0         
_________________________________________________________________
conv5 (Conv2D)               (None, 4000, 1, 32)       32800     
_________________________________________________________________
conv5-BN (BatchNormalization (None, 4000, 1, 32)       128       
_________________________________________________________________
conv5-maxpool (MaxPooling2D) (None, 2000, 1, 32)       0         
_________________________________________________________________
conv5-dropout (Dropout)      (None, 2000, 1, 32)       0         
_________________________________________________________________
conv6 (Conv2D)               (None, 2000, 1, 64)       131136    
_________________________________________________________________
conv6-BN (BatchNormalization (None, 2000, 1, 64)       256       
_________________________________________________________________
conv6-maxpool (MaxPooling2D) (None, 1000, 1, 64)       0         
_________________________________________________________________
conv6-dropout (Dropout)      (None, 1000, 1, 64)       0         
_________________________________________________________________
classifier (Conv2D)          (None, 1000, 1, 360)      23400     
_________________________________________________________________
reshape (Reshape)            (None, 1000, 360)         0         
=================================================================
Total params: 417,976
Trainable params: 417,432
Non-trainable params: 544
_________________________________________________________________
"""