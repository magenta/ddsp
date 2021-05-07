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

N.B. (jesseengel): I tried to make a nice base class. I tried both with multiple
inheritance, and encapsulation, but restoring model parameters seems a bit
fragile given that TF implicitly uses the Python object model for checkpoints,
so I decided to opt for code duplication to make things more robust and preserve
the python object model structure of the original ddsp.training models.

That said, inference models should satisfy the following interface.

Interface:
  Initialize from checkpoint: `model = InferenceModel(ckpt_path)`
  Create SavedModel: `model.save_model(save_dir)`

Need to use model.save_model() as can't override keras model.save().
"""

import ddsp
from ddsp.training import models
from ddsp.training import train_util
import gin
import tensorflow as tf


def parse_operative_config(ckpt_dir):
  with gin.unlock_config():
    operative_config = train_util.get_latest_operative_config(ckpt_dir)
    print(f'Parsing from operative_config {operative_config}')
    gin.parse_config_file(operative_config, skip_unknown=True)


@gin.configurable
class AutoencoderInference(models.Autoencoder):
  """Create an inference-only version of the model."""

  def __init__(self,
               ckpt,
               length_seconds=4,
               remove_reverb=True,
               verbose=True,
               **kwargs):
    self.length_seconds = length_seconds
    self.remove_reverb = remove_reverb
    self.configure_gin(ckpt)
    super().__init__(**kwargs)
    self.restore(ckpt, verbose=verbose)
    self.build_network()

  def configure_gin(self, ckpt):
    """Parse the model operative config to infer new length parameters."""
    parse_operative_config(ckpt)

    # Get preprocessor_type,
    ref = gin.query_parameter('Autoencoder.preprocessor')
    self.preprocessor_type = ref.config_key[-1].split('.')[-1]

    # Get hop_size, and sample_rate from gin config.
    self.sample_rate = gin.query_parameter('Harmonic.sample_rate')
    n_samples_train = gin.query_parameter('Harmonic.n_samples')
    time_steps_train = gin.query_parameter(
        f'{self.preprocessor_type}.time_steps')
    self.hop_size = n_samples_train // time_steps_train

    # Get new lengths for inference.
    self.n_frames = int(self.length_seconds * self.sample_rate / self.hop_size)
    self.n_samples = self.n_frames * self.hop_size
    print('N_Samples:', self.n_samples)
    print('Hop Size:', self.hop_size)
    print('N_Frames:', self.n_frames)

    # Set gin config to new lengths from model properties.
    config = [
        f'Harmonic.n_samples = {self.n_samples}',
        f'FilteredNoise.n_samples = {self.n_samples}',
        f'{self.preprocessor_type}.time_steps = {self.n_frames}',
        'oscillator_bank.use_angular_cumsum = True',
    ]
    if self.remove_reverb:
      # Remove reverb processor.
      processor_group_string = """ProcessorGroup.dag = [
      (@synths.Harmonic(),
        ['amps', 'harmonic_distribution', 'f0_hz']),
      (@synths.FilteredNoise(),
        ['noise_magnitudes']),
      (@processors.Add(),
        ['filtered_noise/signal', 'harmonic/signal']),
      ]"""
      config.append(processor_group_string)

    with gin.unlock_config():
      gin.parse_config(config)

  def save_model(self, save_dir):
    """Saves a SavedModel after initialization."""
    self.save(save_dir)

  def build_network(self):
    """Run a fake batch through the network."""
    db_key = 'power_db' if 'Power' in self.preprocessor_type else 'loudness_db'
    input_dict = {
        db_key: tf.zeros([self.n_frames]),
        'f0_hz': tf.zeros([self.n_frames]),
    }
    # Recursive print of shape.
    print('Inputs to Model:', ddsp.core.map_shape(input_dict))
    unused_outputs = self(input_dict)
    print('Outputs from Model:', ddsp.core.map_shape(unused_outputs))

  @tf.function
  def call(self, inputs, **unused_kwargs):
    """Run the core of the network, get predictions."""
    inputs = ddsp.core.copy_if_tf_function(inputs)
    return super().call(inputs, training=False)


@gin.configurable
class StreamingF0PwInference(models.Autoencoder):
  """Create an inference-only version of the model."""

  def __init__(self, ckpt, verbose=True, **kwargs):
    self.configure_gin(ckpt)
    super().__init__(**kwargs)
    self.restore(ckpt, verbose=verbose)
    self.build_network()

  def configure_gin(self, ckpt):
    """Parse the model operative config with special streaming parameters."""
    parse_operative_config(ckpt)

    # Set streaming specific params.
    time_steps = gin.query_parameter('F0PowerPreprocessor.time_steps')
    n_samples = gin.query_parameter('Harmonic.n_samples')
    samples_per_frame = int(n_samples / time_steps)
    config = [
        'F0PowerPreprocessor.time_steps = 1',
        f'Harmonic.n_samples = {samples_per_frame}',
        f'FilteredNoise.n_samples = {samples_per_frame}',
    ]

    # Remove reverb processor.
    processor_group_string = """ProcessorGroup.dag = [
    (@synths.Harmonic(),
      ['amps', 'harmonic_distribution', 'f0_hz']),
    (@synths.FilteredNoise(),
      ['noise_magnitudes']),
    (@processors.Add(),
      ['filtered_noise/signal', 'harmonic/signal']),
    ]"""
    config.append(processor_group_string)

    with gin.unlock_config():
      gin.parse_config(config)

  def save_model(self, save_dir):
    """Saves a SavedModel after initialization."""
    self.save(save_dir)

  def build_network(self):
    """Run a fake batch through the network."""
    input_dict = {
        'f0_hz': tf.zeros([1]),
        'power_db': tf.zeros([1]),
    }
    print('Inputs to Model:', ddsp.core.map_shape(input_dict))
    unused_outputs = self(input_dict)
    print('Outputs from Model:', ddsp.core.map_shape(unused_outputs))

  @tf.function
  def call(self, inputs, **unused_kwargs):
    """Convert f0 and loudness to synthesizer parameters."""
    inputs = ddsp.core.copy_if_tf_function(inputs)
    controls = super().call(inputs, training=False)
    amps = controls['harmonic']['controls']['amplitudes']
    hd = controls['harmonic']['controls']['harmonic_distribution']
    noise = controls['filtered_noise']['controls']['magnitudes']
    outputs = {
        'amps': amps,
        'hd': hd,
        'noise': noise,
    }
    return outputs


