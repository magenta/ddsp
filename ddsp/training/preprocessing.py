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
"""Library of preprocess functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import ddsp
import gin
import tensorflow.compat.v1 as tf


# ---------------------- Preprocess Helpers ------------------------------------
def resample(x, time_steps):
  """Converts tensor [batch, n_time_in] to [batch, n_time_out, 1]."""
  return ddsp.core.resample(x[:, :, tf.newaxis], time_steps)


# ---------------------- Preprocess objects ------------------------------------
class Preprocessor(object):
  """Base class for chaining a series of preprocessing functions."""

  def __init__(self):
    pass

  def __call__(self, features, training=True):
    return self.get_outputs(features, training)

  def _apply(self, fn, conditioning, keys, **kwargs):
    """Apply preprocessing function `fn` to specific keys.

    Args:
      fn: preprocessing function
      conditioning:  dict of features
      keys: key in `conditioning` dict to apply preprocessing function `fn`
      **kwargs: arguments specific to defined `fn`

    Returns:
      conditioning: dict of transformed features
    """
    for k in keys:
      conditioning[k] = fn(conditioning[k], **kwargs)
    return conditioning

  def get_outputs(self, features, training):
    """Get outputs after preprocessing functions.

    Args:
      features: dict of feature key and tensors
      training: boolean for controlling training-specfic preprocessing
        behavior

    Returns:
      conditioning: dict of transformed features
    """
    raise NotImplementedError


@gin.configurable
class DefaultPreprocessor(Preprocessor):
  """Default class that resamples features and adds `f0_hz` key at end of chain."""

  def __init__(self, time_steps=1000):
    super(DefaultPreprocessor, self).__init__()
    self.time_steps = time_steps

  def _default_processing(self, conditioning):
    """Always resample to `time_steps` and add `f0_hz` key."""
    self._apply(
        resample, conditioning, ('loudness', 'f0'), time_steps=self.time_steps)
    conditioning['f0_hz'] = ddsp.core.midi_to_hz(conditioning['f0'] * 127.0)
    return conditioning

  def get_outputs(self, features, training):
    conditioning = copy.copy(features)
    return self._default_processing(conditioning)

