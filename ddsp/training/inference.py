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
"""Constructs inference version of the models.

These models can be stored as SavedModels by calling model.save() and used
just like other SavedModels.
"""

import os

import ddsp
from ddsp.training import models
from ddsp.training import train_util
import gin
import tensorflow as tf


class AutoencoderInference(models.Autoencoder):
  """Create an inference-only version of the model."""

  @tf.function
  def call(self, input_dict):
    """Run the core of the network, get predictions."""
    input_dict = ddsp.core.copy_if_tf_function(input_dict)
    return super().call(input_dict, training=False)


class StreamingF0Pw(models.Autoencoder):
  """Create an inference-only version of the model."""

  def __init__(self, ckpt, **kwargs):
    self.parse_and_modify_gin_config(ckpt)
    super().__init__(**kwargs)
    self.restore(ckpt)
    self.build_network()

  def parse_and_modify_gin_config(self, ckpt):
    """Parse the model operative config with special streaming parameters."""
    with gin.unlock_config():
      ckpt_dir = os.path.dirname(ckpt)
      operative_config = train_util.get_latest_operative_config(ckpt_dir)
      print(f'Parsing from operative_config {operative_config}')
      gin.parse_config_file(operative_config, skip_unknown=True)
      # Set streaming specific params.
      # Remove reverb processor.
      pg_string = """ProcessorGroup.dag = [
      (@synths.Harmonic(),
        ['amps', 'harmonic_distribution', 'f0_hz']),
      (@synths.FilteredNoise(),
        ['noise_magnitudes']),
      (@processors.Add(),
        ['filtered_noise/signal', 'harmonic/signal']),
      ]"""
      time_steps = gin.query_parameter('F0PowerPreprocessor.time_steps')
      n_samples = gin.query_parameter('Harmonic.n_samples')
      samples_per_frame = int(n_samples / time_steps)
      gin.parse_config([
          'F0PowerPreprocessor.time_steps=1',
          f'Harmonic.n_samples={samples_per_frame}',
          f'FilteredNoise.n_samples={samples_per_frame}',
          pg_string,
      ])

  def build_network(self):
    """Run a fake batch through the network."""
    input_dict = {
        'f0_hz': tf.zeros([1]),
        'power_db': tf.zeros([1]),
    }
    print('Inputs to Model:', input_dict)
    unused_outputs = self(input_dict)
    print('Outputs', unused_outputs)

  @tf.function
  def call(self, input_dict):
    """Convert f0 and loudness to synthesizer parameters."""
    input_dict = ddsp.core.copy_if_tf_function(input_dict)
    controls = super().call(input_dict, training=False)
    amps = controls['harmonic']['controls']['amplitudes']
    hd = controls['harmonic']['controls']['harmonic_distribution']
    noise = controls['filtered_noise']['controls']['magnitudes']
    return amps, hd, noise

