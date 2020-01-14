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

hz_to_midi = ddsp.core.hz_to_midi
F0_RANGE = ddsp.spectral_ops.F0_RANGE
LD_RANGE = ddsp.spectral_ops.LD_RANGE


# ---------------------- Preprocess Helpers ------------------------------------
def at_least_3d(x):
  """Adds a channel dimension."""
  return x[:, :, tf.newaxis] if len(x.shape) == 2 else x


# ---------------------- Preprocess objects ------------------------------------
class Preprocessor(object):
  """Base class for chaining a series of preprocessing functions."""

  def __init__(self):
    pass

  def __call__(self, features, training=True):
    """Get outputs after preprocessing functions.

    Args:
      features: dict of feature key and tensors
      training: boolean for controlling training-specfic preprocessing behavior

    Returns:
      Dictionary of transformed features
    """
    return copy.copy(features)


@gin.register
class DefaultPreprocessor(Preprocessor):
  """Default class that resamples features and adds `f0_hz` key."""

  def __init__(self, time_steps=1000):
    super(DefaultPreprocessor, self).__init__()
    self.time_steps = time_steps

  def __call__(self, features, training=True):
    super(DefaultPreprocessor, self).__call__(features, training)
    return self._default_processing(features)

  def _default_processing(self, features):
    """Always resample to `time_steps` and scale 'loudness_db' and 'f0_hz'."""
    for k in ['loudness_db', 'f0_hz']:
      features[k] = ddsp.core.resample(features[k], n_timesteps=self.time_steps)
      features[k] = at_least_3d(features[k])
    # For NN training, scale frequency and loudness to the range [0, 1].
    # Log-scale f0 features. Loudness from [-1, 0] to [1, 0].
    features['f0_scaled'] = hz_to_midi(features['f0_hz']) / F0_RANGE
    features['ld_scaled'] = (features['loudness_db'] / LD_RANGE) + 1.0
    return features


