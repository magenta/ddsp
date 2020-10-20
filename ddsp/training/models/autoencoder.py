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
"""Model that outputs coefficeints of an additive synthesizer."""

import ddsp
from ddsp.training.models.model import Model


class Autoencoder(Model):
  """Wrap the model function for dependency injection with gin."""

  def __init__(self,
               preprocessor=None,
               encoder=None,
               decoder=None,
               processor_group=None,
               losses=None,
               name='autoencoder'):
    super().__init__(name=name)
    self.preprocessor = preprocessor
    self.encoder = encoder
    self.decoder = decoder
    self.processor_group = processor_group
    self.loss_objs = ddsp.core.make_iterable(losses)

  def controls_to_audio(self, controls):
    return controls[self.processor_group.name]['signal']

  def encode(self, features, training=True):
    """Get conditioning by preprocessing then encoding."""
    if self.preprocessor is not None:
      conditioning = self.preprocessor(features, training=training)
    else:
      conditioning = features
    return conditioning if self.encoder is None else self.encoder(conditioning)

  def decode(self, conditioning, training=True):
    """Get generated audio by decoding than processing."""
    processor_inputs = self.decoder(conditioning, training=training)
    return self.processor_group(processor_inputs)

  def call(self, features, training=True):
    """Run the core of the network, get predictions and loss."""
    conditioning = self.encode(features, training=training)
    audio_gen = self.decode(conditioning, training=training)
    if training:
      for loss_obj in self.loss_objs:
        loss = loss_obj(features['audio'], audio_gen)
        self._losses_dict[loss_obj.name] = loss
    return audio_gen

  def get_controls(self, features, keys=None, training=False):
    """Returns specific processor_group controls."""
    conditioning = self.encode(features, training=training)
    processor_inputs = self.decoder(conditioning)
    controls = self.processor_group.get_controls(processor_inputs)
    # Also build on get_controls(), instead of just __call__().
    self.built = True
    # If wrapped in tf.function, only calculates keys of interest.
    return controls if keys is None else {k: controls[k] for k in keys}
