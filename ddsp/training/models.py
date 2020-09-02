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

import functools
import time

from absl import logging
import ddsp
from ddsp.training import train_util
import gin
import tensorflow.compat.v2 as tf

tfkl = tf.keras.layers


@gin.configurable
def get_model(model=gin.REQUIRED):
  """Gin configurable function get a 'global' model for use in ddsp_run.py.

  Convenience for using the same model in train(), evaluate(), and sample().
  Args:
    model: An instantiated model, such as 'models.Autoencoder()'.

  Returns:
    The 'global' model specifieed in the gin config.
  """
  return model


class Model(tf.keras.Model):
  """Wrap the model function for dependency injection with gin."""

  def __init__(self, name='model'):
    super().__init__(name=name)
    self._losses_dict = {}

  def __call__(self, *args, return_losses=False, **kwargs):
    """Reset the losses dict on each call.

    Args:
      *args: Arguments passed on to call().
      return_losses: Return a dictionary of losses in addition to the call()
        function returns.
      **kwargs: Keyword arguments passed on to call().

    Returns:
      Function results if return_losses=False, else the function results
        and a dictionary of losses, {loss_name: loss_value}.
    """
    self._losses_dict = {}
    results = super().__call__(*args, **kwargs)
    if not return_losses:
      return results
    else:
      self._losses_dict['total_loss'] = tf.reduce_sum(
          list(self._losses_dict.values()))
      return results, self._losses_dict

  def restore(self, checkpoint_path):
    """Restore model and optimizer from a checkpoint."""
    start_time = time.time()
    latest_checkpoint = train_util.get_latest_chekpoint(checkpoint_path)
    if latest_checkpoint is not None:
      checkpoint = tf.train.Checkpoint(model=self)
      checkpoint.restore(latest_checkpoint).expect_partial()
      logging.info('Loaded checkpoint %s', latest_checkpoint)
      logging.info('Loading model took %.1f seconds', time.time() - start_time)
    else:
      logging.info('Could not find checkpoint to load at %s, skipping.',
                   checkpoint_path)

  def get_controls(self, features, keys=None, training=False):
    """Base method for getting controls. Not implemented."""
    raise NotImplementedError('`get_controls` not implemented in base class!')


@gin.configurable
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
    conditioning = self.preprocessor(features, training=training)
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


@gin.configurable
class TranscribingAutoencoder(Model):
  """Multi-level Transcribing Autoencoder (TAE).

  EXPERIMENTAL

  Builds a hierarchical model of audio:
  Audio -> Sinusoid -> Harmonic -> Midi -> Harmonic -> Sinusoid -> Audio.

  Unlike other models, the processor group is defined in the constructor, as
  there are many synthesizer specific regularizers and self-supervised losses.
  """

  def __init__(self,
               # Encoders.
               sinusoidal_encoder=None,
               harmonic_encoder=None,
               # Losses.
               losses=None,
               sinusoidal_consistency_losses=None,
               harmonic_consistency_losses=None,
               filtered_noise_consistency_loss=None,
               twm_loss=None,
               harmonic_distribution_prior=None,
               # Processor Group.
               freq_scale_fn=None,
               reverb=True,
               n_samples=64000,
               sample_rate=16000,
               # Training.
               stop_gradient=True,
               name='multi_level_transcribing_autoencoder'):
    """Constructor."""
    super().__init__(name=name)
    # Network objects.
    self.sinusoidal_encoder = sinusoidal_encoder
    self.harmonic_encoder = harmonic_encoder
    # Losses.
    self.audio_loss_objs = ddsp.core.make_iterable(losses)
    self.sinusoidal_consistency_losses = ddsp.core.make_iterable(
        sinusoidal_consistency_losses)
    self.harmonic_consistency_losses = ddsp.core.make_iterable(
        harmonic_consistency_losses)
    self.filtered_noise_consistency_loss = filtered_noise_consistency_loss
    self.twm_loss = twm_loss
    self.harmonic_distribution_prior = harmonic_distribution_prior
    # Training.
    self.stop_gradient = stop_gradient

    # ==============
    # ProcessorGroup
    # ==============
    # Define processor group in the constructor for more flexible routing.
    self.n_samples = n_samples
    self.sample_rate = sample_rate

    self.amps_scale_fn = ddsp.core.exp_sigmoid

    self.freq_scale_fn = freq_scale_fn or functools.partial(
        ddsp.core.frequencies_softmax, depth=64)

    self.sinusoidal_synth = ddsp.synths.Sinusoidal(
        n_samples=self.n_samples,
        sample_rate=self.sample_rate,
        amp_scale_fn=None,
        freq_scale_fn=None,
        name='sinusoidal')

    self.filtered_noise_synth = ddsp.synths.FilteredNoise(
        n_samples=self.n_samples,
        window_size=0,
        scale_fn=None,
        name='filtered_noise')

    dag = [
        (self.sinusoidal_synth,
         ['amplitudes', 'frequencies']),
        (self.filtered_noise_synth,
         ['noise_magnitudes']),
        (ddsp.processors.Add(),
         [f'{self.filtered_noise_synth.name}/signal',
          f'{self.sinusoidal_synth.name}/signal']),
    ]

    if reverb:
      self.reverb = ddsp.effects.FilteredNoiseReverb(
          reverb_length=int(self.sample_rate * 2),
          window_size=257,
          n_frames=500,
          n_filter_banks=16,
          trainable=True,
          name='reverb')
      dag.append((self.reverb, ['add/signal']))

    self.processor_group = ddsp.processors.ProcessorGroup(dag=dag)

  def get_controls(self, features, keys=None, training=False):
    """Returns specific processor_group controls."""
    # For now just use the real data.
    if isinstance(features, (list, tuple)):
      features, unused_ss_features = self.parse_zipped_features(features)

    # Encode the data from audio to sinusoids.
    pg_in = self.sinusoidal_encoder(features, training=training)

    # Manually apply the scaling nonlinearities.
    pg_in['frequencies'] = self.freq_scale_fn(pg_in['frequencies'])
    pg_in['amplitudes'] = self.amps_scale_fn(pg_in['amplitudes'])
    pg_in['noise_magnitudes'] = self.amps_scale_fn(pg_in['noise_magnitudes'])
    controls = self.processor_group.get_controls(pg_in)

    # Append normal training procedure outputs.
    outputs = self.forward(features, training)
    controls.update(outputs)

    self.built = True
    # If wrapped in tf.function, only calculates keys of interest.
    return controls if keys is None else {k: controls[k] for k in keys}

  def generate_synthetic_audio(self, features):
    """Convert synthetic controls into audio."""
    return self.processor_group({
        'amplitudes': features['sin_amps'],
        'frequencies': features['sin_freqs'],
        'noise_magnitudes': features['noise_magnitudes']
    })

  def parse_zipped_features(self, features):
    """Extract self-supervised dictionary from list with normal features.

    Args:
      features: A list/tuple of feature dictionaries.

    Returns:
      An ordered tuple of feature dictionaries (normal, self-supervised).
    """
    # Assumes two datasets, supervised, self_supervised (has 'sin_amps' key).
    assert len(features) == 2
    ss_idx = int(features[1].get('sin_amps') is not None)
    s_idx = int(not ss_idx)
    return features[s_idx], features[ss_idx]

  def call(self, features, training=True):
    """Run the core of the network, get predictions and loss."""
    if isinstance(features, (list, tuple)):
      # Train on both normal and self-supervised data.
      features, ss_features = self.parse_zipped_features(features)

      # Generate audio from controls if self_supervised.
      ss_features = ddsp.core.copy_if_tf_function(ss_features)
      ss_features['audio'] = self.generate_synthetic_audio(ss_features)

      # Concatenate inputs for more efficiency.
      batch_size = features['audio'].shape[0]
      inputs = {'audio': tf.concat([features['audio'],
                                    ss_features['audio']], axis=0)}

      # Pass through the model.
      all_outputs = self.forward(inputs, training)

      # Split outputs.
      outputs = {k: v[:batch_size] for k, v in all_outputs.items()}
      ss_outputs = {k: v[batch_size:] for k, v in all_outputs.items()}

      # Compute losses.
      self.append_losses(outputs)
      self.append_losses(ss_outputs, ss_features)

    elif features.get('sin_amps') is not None:
      # Self supervised pretraining.
      ss_features = ddsp.core.copy_if_tf_function(features)
      ss_features['audio'] = self.generate_synthetic_audio(ss_features)
      outputs = self.forward(ss_features, training)
      self.append_losses(outputs)
      self.append_losses(outputs, ss_features)

    else:
      # Normal training procedure.
      outputs = self.forward(features, training)
      self.append_losses(outputs)

    if self.harmonic_encoder is not None:
      return outputs['harm_audio']
    else:
      return outputs['sin_audio']

  def forward(self, features, training=True):
    """Run forward pass of model (no losses) on a dictionary of features."""
    # Audio -> Sinusoids -------------------------------------------------------
    audio = features['audio']

    # Encode the data from audio to sinusoids.
    pg_in = self.sinusoidal_encoder(features, training=training)

    # Manually apply the scaling nonlinearities.
    sin_freqs = self.freq_scale_fn(pg_in['frequencies'])
    sin_amps = self.amps_scale_fn(pg_in['amplitudes'])
    noise_magnitudes = self.amps_scale_fn(pg_in['noise_magnitudes'])
    pg_in['frequencies'] = sin_freqs
    pg_in['amplitudes'] = sin_amps
    pg_in['noise_magnitudes'] = noise_magnitudes

    # Reconstruct sinusoidal audio.
    sin_audio = self.processor_group(pg_in)

    outputs = {
        # Input signal.
        'audio': audio,
        # Filtered noise signal.
        'noise_magnitudes': noise_magnitudes,
        # Sinusoidal signal.
        'sin_audio': sin_audio,
        'sin_amps': sin_amps,
        'sin_freqs': sin_freqs,
    }

    # Sinusoids -> Harmonics ---------------------------------------------------
    # Encode the sinusoids into a harmonics.
    if self.stop_gradient:
      sin_freqs = tf.stop_gradient(sin_freqs)
      sin_amps = tf.stop_gradient(sin_amps)
      noise_magnitudes = tf.stop_gradient(noise_magnitudes)

    if self.harmonic_encoder is not None:
      harm_amp, harm_dist, f0_hz = self.harmonic_encoder(sin_freqs, sin_amps)

      # Decode harmonics back to sinusoids.
      n_harmonics = int(harm_dist.shape[-1])
      harm_freqs = ddsp.core.get_harmonic_frequencies(f0_hz, n_harmonics)
      harm_amps = harm_amp * harm_dist

      # Reconstruct harmonic audio.
      pg_in['frequencies'] = harm_freqs
      pg_in['amplitudes'] = harm_amps
      pg_in['noise_magnitudes'] = noise_magnitudes
      harm_audio = self.processor_group(pg_in)

      outputs.update({
          # Harmonic signal.
          'harm_audio': harm_audio,
          'harm_amp': harm_amp,
          'harm_dist': harm_dist,
          'f0_hz': f0_hz,
          # Harmonic Sinusoids.
          'harm_freqs': harm_freqs,
          'harm_amps': harm_amps,
      })

    return outputs

  def append_losses(self, outputs, self_supervised_features=None):
    """Compute losses from outputs and append to self._losses_dict."""
    # Aliases.
    o = outputs
    f = self_supervised_features

    # Unsupervised losses.
    if f is None:
      # Sinusoidal autoencoder loss.
      for loss_obj in self.audio_loss_objs:
        name = 'sin_{}'.format(loss_obj.name)
        self._losses_dict[name] = loss_obj(o['audio'], o['sin_audio'])

      if self.harmonic_encoder is not None:
        # Add prior regularization on harmonic distribution.
        hdp = self.harmonic_distribution_prior
        if hdp is not None:
          self._losses_dict.update({hdp.name: hdp(o['harm_dist'])})

        # Harmonic autoencoder loss.
        for loss_obj in self.audio_loss_objs:
          name = 'harm_{}'.format(loss_obj.name)
          self._losses_dict[name] = loss_obj(o['audio'], o['harm_audio'])

        # Sinusoidal<->Harmonic consistency loss.
        if self.sinusoidal_consistency_losses:
          sin_amps = o['sin_amps']
          sin_freqs = o['sin_freqs']
          if self.stop_gradient:
            # Don't propagate harmonic errors to sinusoidal predictions.
            sin_amps = tf.stop_gradient(sin_amps)
            sin_freqs = tf.stop_gradient(sin_freqs)
          for loss_obj in self.sinusoidal_consistency_losses:
            self._losses_dict[loss_obj.name] = loss_obj(
                sin_amps, sin_freqs, o['harm_amps'], o['harm_freqs'])

      # Two-way mismatch loss between sinusoids and harmonics.
      if self.twm_loss is not None:
        if self.harmonic_encoder is not None:
          loss = self.twm_loss(o['f0_hz'], o['sin_freqs'], o['sin_amps'])
        else:
          loss = self.twm_loss(o['sin_freqs'], o['sin_freqs'], o['sin_amps'])
        self._losses_dict[self.twm_loss.name] = loss

    # Self-supervised Losses.
    else:
      # Sinusoidal self-supervision.
      if self.sinusoidal_consistency_losses:
        for loss_obj in self.sinusoidal_consistency_losses:
          name = 'ss_' + loss_obj.name
          self._losses_dict[name] = loss_obj(
              o['sin_amps'], o['sin_freqs'], f['sin_amps'], f['sin_freqs'])

      # Filtered noise self-supervision.
      fncl = self.filtered_noise_consistency_loss
      if fncl is not None:
        name = 'ss_' + fncl.name
        self._losses_dict[name] = fncl(o['noise_magnitudes'],
                                       f['noise_magnitudes'])

      # Harmonic self-supervision.
      if self.harmonic_consistency_losses:
        for loss_obj in self.harmonic_consistency_losses:
          if isinstance(loss_obj, ddsp.losses.HarmonicConsistencyLoss):
            # L1 loss of harmonic synth controls.
            losses = loss_obj(o['harm_amp'], f['harm_amp'],
                              o['harm_dist'], f['harm_dist'],
                              o['f0_hz'], f['f0_hz'])
            losses = {'ss_' + k: v for k, v in losses.items()}
            self._losses_dict.update(losses)
          else:
            # Same consistency loss as sinusoidal models.
            name = 'ss_harm_' + loss_obj.name
            self._losses_dict[name] = loss_obj(
                o['harm_amp'], o['f0_hz'], f['harm_amp'], f['f0_hz'])


