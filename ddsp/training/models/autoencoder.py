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
"""Model that encodes audio features and decodes with a ddsp processor group."""

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
               **kwargs):
    super().__init__(**kwargs)
    self.preprocessor = preprocessor
    self.encoder = encoder
    self.decoder = decoder
    self.processor_group = processor_group
    self.loss_objs = ddsp.core.make_iterable(losses)

  def encode(self, features, training=True):
    """Get conditioning by preprocessing then encoding."""
    if self.preprocessor is not None:
      conditioning = self.preprocessor(features, training=training)
    else:
      conditioning = ddsp.core.copy_if_tf_function(features)
    if self.encoder is not None:
      encoder_out = self.encoder(conditioning)
      conditioning.update(encoder_out)
    return conditioning

  def decode(self, conditioning, training=True):
    """Get generated audio by decoding than processing."""
    pg_in = self.decoder(conditioning, training=training)
    pg_in.update(conditioning)
    return self.processor_group(pg_in)

  def get_audio_from_outputs(self, outputs):
    """Extract audio output tensor from outputs dict of call()."""
    return self.processor_group.get_signal(outputs)

  def call(self, features, training=True):
    """Run the core of the network, get predictions and loss."""
    conditioning = self.encode(features, training=training)
    pg_in = self.decoder(conditioning, training=training)
    pg_in.update(conditioning)
    outputs = self.processor_group.get_controls(pg_in)
    outputs['audio_synth'] = self.processor_group.get_signal(outputs)
    if training:
      self._update_losses_dict(
          self.loss_objs, features['audio'], outputs['audio_synth'])
    return outputs

