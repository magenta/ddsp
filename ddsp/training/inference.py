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
"""Constructs inference version of the models.

These models can be stored as SavedModels by calling model.save() and used
just like other SavedModels.
"""

from ddsp.training import models
import tensorflow.compat.v2 as tf


class AutoencoderInference(models.Autoencoder):
  """Create an inference-only version of the model."""

  @tf.function
  def call(self, features):
    """Run the core of the network, get predictions and loss."""
    return super().call(features, training=False)
