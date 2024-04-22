# Copyright 2024 The DDSP Authors.
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


class VSTBaseModule(models.Autoencoder):
  """VST inference modules, for models trained with `models/vst/vst.gin`."""

  def __init__(self,
               ckpt,
               verbose=False,
               **kwargs):
    self.parse_gin(ckpt)
    self.configure_gin()
    super().__init__(**kwargs)
    self.restore(ckpt, verbose=verbose)
    self.build_network()

  def parse_gin(self, ckpt):
    """Parse the model operative config with special streaming parameters."""
    parse_operative_config(ckpt)

    # Get Frame Size / Hop Size.
    self.frame_size = gin.query_parameter('%frame_size')
    frame_rate = gin.query_parameter('%frame_rate')
    self.sample_rate = gin.query_parameter('%sample_rate')
    self.hop_size = self.sample_rate // frame_rate

    # Get number of outputs.
    output_splits = dict(gin.query_parameter('RnnFcDecoder.output_splits'))
    self.n_harmonics = output_splits['harmonic_distribution']
    self.n_noise = output_splits['noise_magnitudes']

    # Get RNN dimesnions.
    self.state_size = gin.query_parameter('RnnFcDecoder.rnn_channels')

    # Get interpolation method.
    self.resample_method = gin.query_parameter('Harmonic.amp_resample_method')

  def configure_gin(self):
    raise NotImplementedError

  def restore(self, checkpoint_path, verbose=True):
    # Leave out preprocessor to allow loading different CREPE models.
    restore_keys = ['decoder']
    super().restore(checkpoint_path, verbose=verbose, restore_keys=restore_keys)

  def save_model(self, save_dir):
    """Saves a SavedModel after initialization."""
    # self.save(save_dir)
    tf.saved_model.save(self, save_dir, signatures=self._signatures)

  @property
  def _signatures(self):
    raise NotImplementedError

  def _build_network(self, *dummy_inputs):
    """Helper function to build the network with dummy input args."""
    print('Inputs to Model:', ddsp.core.map_shape(dummy_inputs))
    unused_outputs = self(*dummy_inputs)
    print('Outputs from Model:', ddsp.core.map_shape(unused_outputs))

  def call(self):
    raise NotImplementedError


class VSTExtractFeatures(VSTBaseModule):
  """VST inference modules, for models trained with `models/vst/vst.gin`."""

  def __init__(self,
               ckpt,
               crepe_saved_model_path=None,
               **kwargs):
    self.crepe_saved_model_path = crepe_saved_model_path
    super().__init__(ckpt, **kwargs)

  def configure_gin(self):
    """Parse the model operative config with special streaming parameters."""
    # Customize config.
    config = [
        'OnlineF0PowerPreprocessor.padding = "valid"',
        'OnlineF0PowerPreprocessor.compute_f0 = True',
        'OnlineF0PowerPreprocessor.compute_power = True',
        'OnlineF0PowerPreprocessor.viterbi = False',
    ]
    if self.crepe_saved_model_path is not None:
      config.append('OnlineF0PowerPreprocessor.crepe_saved_model_path = '
                    f'\'{self.crepe_saved_model_path}\'')
    with gin.unlock_config():
      gin.parse_config(config)

  @property
  def _signatures(self):
    return {'call': self.call.get_concrete_function(
        audio=tf.TensorSpec(shape=[self.frame_size], dtype=tf.float32)
        )}

  def build_network(self):
    """Run a fake batch through the network."""
    # Need two frames because of interpolation.
    audio = tf.zeros([self.frame_size])
    self._build_network(audio)

  @tf.function
  def call(self, audio):
    """Convert f0 and loudness to synthesizer parameters."""
    audio = tf.reshape(audio, [1, self.frame_size])

    inputs = {
        'audio': audio,
        'f0_hz': tf.zeros([1, 1]),  # Dummy.
        'f0_confidence': tf.zeros([1, 1]),  # Dummy.
    }
    outputs = self.preprocessor(inputs)

    # Return 1-D tensors.
    # All shapes are [1, 1, 1].
    f0_hz = outputs['f0_hz'][0, 0]
    f0_scaled = outputs['f0_scaled'][0, 0]
    pw_db = outputs['pw_db'][0, 0]
    pw_scaled = outputs['pw_scaled'][0, 0]
    return f0_hz, f0_scaled, pw_db, pw_scaled


class VSTPredictControls(VSTBaseModule):
  """VST inference modules, for models trained with `models/vst/vst.gin`."""

  def configure_gin(self):
    """Parse the model operative config with special streaming parameters."""
    pass

  @property
  def _signatures(self):
    return {'call': self.call.get_concrete_function(
        f0_scaled=tf.TensorSpec(shape=[1], dtype=tf.float32),
        pw_scaled=tf.TensorSpec(shape=[1], dtype=tf.float32),
    )}

  def build_network(self):
    """Run a fake batch through the network."""
    # Need two frames because of interpolation.
    f0_scaled = tf.zeros([1])
    pw_scaled = tf.zeros([1])
    self._build_network(f0_scaled, pw_scaled)

  @tf.function
  def call(self, f0_scaled, pw_scaled):
    """Convert f0 and loudness to synthesizer parameters."""
    f0_scaled = tf.reshape(f0_scaled, [1, 1, 1])
    pw_scaled = tf.reshape(pw_scaled, [1, 1, 1])

    f0_hz = ddsp.training.preprocessing.inv_scale_f0_hz(f0_scaled)

    inputs = {
        'f0_scaled': f0_scaled,
        'pw_scaled': pw_scaled,
    }

    # Run through the model.
    outputs = self.decoder(inputs, training=False)

    # Apply the nonlinearities.
    harm_controls = self.processor_group.harmonic.get_controls(
        outputs['amps'], outputs['harmonic_distribution'], f0_hz)

    noise_controls = self.processor_group.filtered_noise.get_controls(
        outputs['noise_magnitudes']
    )

    # Return 1-D tensors.
    amps = harm_controls['amplitudes'][0, 0]
    hd = harm_controls['harmonic_distribution'][0, 0]
    noise = noise_controls['magnitudes'][0, 0]
    return amps, hd, noise


class VSTStatelessPredictControls(VSTBaseModule):
  """Predict VST controls, but explicitly handle RNN state."""

  def configure_gin(self):
    """Parse the model operative config with special streaming parameters."""
    config = [
        'RnnFcDecoder.stateless = True',
    ]
    with gin.unlock_config():
      gin.parse_config(config)

  @property
  def _signatures(self):
    return {'call': self.call.get_concrete_function(
        f0_scaled=tf.TensorSpec(shape=[1], dtype=tf.float32),
        pw_scaled=tf.TensorSpec(shape=[1], dtype=tf.float32),
        state=tf.TensorSpec(shape=[self.state_size], dtype=tf.float32),
    )}

  def build_network(self):
    """Run a fake batch through the network."""
    # Need two frames because of interpolation.
    f0_scaled = tf.zeros([1])
    pw_scaled = tf.zeros([1])
    state = tf.zeros([self.state_size])
    self._build_network(f0_scaled, pw_scaled, state)

  @tf.function
  def call(self, f0_scaled, pw_scaled, state):
    """Convert f0 and loudness to synthesizer parameters."""
    f0_scaled = tf.reshape(f0_scaled, [1, 1, 1])
    pw_scaled = tf.reshape(pw_scaled, [1, 1, 1])
    state = tf.reshape(state, [1, self.state_size])

    f0_hz = ddsp.training.preprocessing.inv_scale_f0_hz(f0_scaled)

    inputs = {
        'f0_scaled': f0_scaled,
        'pw_scaled': pw_scaled,
        'state': state,
    }

    # Run through the model.
    outputs = self.decoder(inputs, training=False)

    # Apply the nonlinearities.
    harm_controls = self.processor_group.harmonic.get_controls(
        outputs['amps'], outputs['harmonic_distribution'], f0_hz)

    noise_controls = self.processor_group.filtered_noise.get_controls(
        outputs['noise_magnitudes']
    )

    # Return 1-D tensors.
    amps = harm_controls['amplitudes'][0, 0]
    hd = harm_controls['harmonic_distribution'][0, 0]
    noise = noise_controls['magnitudes'][0, 0]
    state = outputs['state'][0]
    return amps, hd, noise, state


@gin.configurable
class VSTSynthesize(tf.keras.Model):
  """VST inference modules, for models trained with `models/vst/vst.gin`."""

  def __init__(self,
               ckpt,
               new_hop_size=None,
               **kwargs):
    super().__init__(**kwargs)
    self.new_hop_size = new_hop_size
    self.parse_gin(ckpt)
    self.build_network()

  # Carried over from VSTBaseModule. Need separate class to not include vars.
  def save_model(self, save_dir):
    """Saves a SavedModel after initialization."""
    # self.save(save_dir)
    tf.saved_model.save(self, save_dir, signatures=self._signatures)

  def _build_network(self, *dummy_inputs):
    """Helper function to build the network with dummy input args."""
    print('Inputs to Model:', ddsp.core.map_shape(dummy_inputs))
    unused_outputs = self(*dummy_inputs)
    print('Outputs from Model:', ddsp.core.map_shape(unused_outputs))

  def parse_gin(self, ckpt):
    """Parse the model operative config with special streaming parameters."""
    parse_operative_config(ckpt)

    # Get Frame Size / Hop Size.
    self.frame_size = gin.query_parameter('%frame_size')
    frame_rate = gin.query_parameter('%frame_rate')
    self.sample_rate = gin.query_parameter('%sample_rate')
    self.hop_size = self.sample_rate // frame_rate

    # Get number of outputs.
    output_splits = dict(gin.query_parameter('RnnFcDecoder.output_splits'))
    self.n_harmonics = output_splits['harmonic_distribution']
    self.n_noise = output_splits['noise_magnitudes']

    # Get interpolation method.
    self.resample_method = gin.query_parameter('Harmonic.amp_resample_method')

    config = [
        'harmonic_oscillator_bank.use_angular_cumsum = True',
    ]
    with gin.unlock_config():
      gin.parse_config(config)

    self.hop_size = self.new_hop_size if self.new_hop_size else self.hop_size
    self.filtered_noise = ddsp.synths.FilteredNoise(
        n_samples=self.hop_size,
        window_size=gin.query_parameter('FilteredNoise.window_size'),
    )

  @property
  def _signatures(self):
    return {'call': self.call.get_concrete_function(
        amps=tf.TensorSpec(shape=[1], dtype=tf.float32),
        prev_amps=tf.TensorSpec(shape=[1], dtype=tf.float32),
        hd=tf.TensorSpec(shape=[self.n_harmonics], dtype=tf.float32),
        prev_hd=tf.TensorSpec(shape=[self.n_harmonics], dtype=tf.float32),
        f0=tf.TensorSpec(shape=[1], dtype=tf.float32),
        prev_f0=tf.TensorSpec(shape=[1], dtype=tf.float32),
        noise=tf.TensorSpec(shape=[self.n_noise], dtype=tf.float32),
        prev_phase=tf.TensorSpec(shape=[1], dtype=tf.float32),
    )}

  def build_network(self):
    """Run a fake batch through the network."""
    # Need two frames because of interpolation.
    amps = tf.zeros([1])
    prev_amps = tf.zeros([1])
    hd = tf.zeros([self.n_harmonics])
    prev_hd = tf.zeros([self.n_harmonics])
    f0 = tf.zeros([1])
    prev_f0 = tf.zeros([1])
    noise = tf.zeros([self.n_noise])
    prev_phase = tf.zeros([1])
    self._build_network(
        amps, prev_amps, hd, prev_hd, f0, prev_f0, noise, prev_phase)

  @tf.function
  def call(self, amps, prev_amps, hd, prev_hd,
           f0, prev_f0, noise, prev_phase):
    """Compute a frame of audio, single example, single frame."""
    # Make 3-D tensors, two frames for interpolation.
    amps = tf.reshape(
        tf.concat([prev_amps[None, :], amps[None, :]], axis=0),
        [1, 2, 1])
    hd = tf.reshape(
        tf.concat([prev_hd[None, :], hd[None, :]], axis=0),
        [1, 2, self.n_harmonics])
    f0 = tf.reshape(
        tf.concat([prev_f0[None, :], f0[None, :]], axis=0),
        [1, 2, 1])
    noise = tf.reshape(
        tf.concat([noise[None, :], noise[None, :]], axis=0),
        [1, 2, self.n_noise])
    prev_phase = tf.reshape(prev_phase, [1, 1, 1])

    harm_audio, final_phase = ddsp.core.streaming_harmonic_synthesis(
        frequencies=f0,
        amplitudes=amps,
        harmonic_distribution=hd,
        initial_phase=prev_phase,
        n_samples=self.hop_size,
        sample_rate=self.sample_rate,
        amp_resample_method=self.resample_method)

    noise_audio = self.filtered_noise.get_signal(noise)
    audio_out = harm_audio + noise_audio

    # Return 1-D outputs.
    audio_out = audio_out[0]
    final_phase = final_phase[0, 0]
    return audio_out, final_phase


@gin.configurable
class VSTSynthesizeHarmonic(VSTSynthesize):
  """VST inference modules, for models trained with `models/vst/vst.gin`."""

  @property
  def _signatures(self):
    return {'call': self.call.get_concrete_function(
        amps=tf.TensorSpec(shape=[1], dtype=tf.float32),
        prev_amps=tf.TensorSpec(shape=[1], dtype=tf.float32),
        hd=tf.TensorSpec(shape=[self.n_harmonics], dtype=tf.float32),
        prev_hd=tf.TensorSpec(shape=[self.n_harmonics], dtype=tf.float32),
        f0=tf.TensorSpec(shape=[1], dtype=tf.float32),
        prev_f0=tf.TensorSpec(shape=[1], dtype=tf.float32),
        prev_phase=tf.TensorSpec(shape=[1], dtype=tf.float32),
    )}

  def build_network(self):
    """Run a fake batch through the network."""
    # Need two frames because of interpolation.
    amps = tf.zeros([1])
    prev_amps = tf.zeros([1])
    hd = tf.zeros([self.n_harmonics])
    prev_hd = tf.zeros([self.n_harmonics])
    f0 = tf.zeros([1])
    prev_f0 = tf.zeros([1])
    prev_phase = tf.zeros([1])
    self._build_network(
        amps, prev_amps, hd, prev_hd, f0, prev_f0, prev_phase)

  @tf.function
  def call(self, amps, prev_amps, hd, prev_hd,
           f0, prev_f0, prev_phase):
    """Compute a frame of audio, single example, single frame."""
    # Make 3-D tensors, two frames for interpolation.
    amps = tf.reshape(
        tf.concat([prev_amps[None, :], amps[None, :]], axis=0),
        [1, 2, 1])
    hd = tf.reshape(
        tf.concat([prev_hd[None, :], hd[None, :]], axis=0),
        [1, 2, self.n_harmonics])
    f0 = tf.reshape(
        tf.concat([prev_f0[None, :], f0[None, :]], axis=0),
        [1, 2, 1])
    prev_phase = tf.reshape(prev_phase, [1, 1, 1])

    audio_out, final_phase = ddsp.core.streaming_harmonic_synthesis(
        frequencies=f0,
        amplitudes=amps,
        harmonic_distribution=hd,
        initial_phase=prev_phase,
        n_samples=self.hop_size,
        sample_rate=self.sample_rate,
        amp_resample_method=self.resample_method)

    # Return 1-D outputs.
    audio_out = audio_out[0]
    final_phase = final_phase[0, 0]
    return audio_out, final_phase


@gin.configurable
class VSTSynthesizeNoise(VSTSynthesize):
  """VST inference modules, for models trained with `models/vst/vst.gin`."""

  @property
  def _signatures(self):
    return {'call': self.call.get_concrete_function(
        noise=tf.TensorSpec(shape=[self.n_noise], dtype=tf.float32),
    )}

  def build_network(self):
    """Run a fake batch through the network."""
    # Need two frames because of interpolation.
    noise = tf.zeros([self.n_noise])
    self._build_network(noise)

  @tf.function
  def call(self, noise):
    """Compute a frame of audio, single example, single frame."""
    # Make 3-D tensors, two frames for interpolation.
    noise = tf.reshape(
        tf.concat([noise[None, :], noise[None, :]], axis=0),
        [1, 2, self.n_noise])

    audio_out = self.filtered_noise.get_signal(noise)

    # Return 1-D outputs.
    audio_out = audio_out[0]
    return audio_out


