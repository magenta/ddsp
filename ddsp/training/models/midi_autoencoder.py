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

"""Model that encodes and decodes from MIDI.

**** EXPERIMENTAL -- Work in Progress. ****
This code is being used for active research. We provide no guarantees! :-P

"""
import ddsp
from ddsp.training import nn
from ddsp.training import preprocessing
from ddsp.training.models.model import Model
import tensorflow as tf


class MidiAutoencoder(Model):
  """Autoencoder that has a MIDI representation at its bottleneck.

  Here is a crude diagram of the MidiAutoencoder architecture:

          MidiEncoder               MidiDecoder
                 +------> (MIDI) ------+
                 |                     |
                 |                     v
             (f0,amps,             (f0,amps,
              hd,noise)---+         hd,noise)
                 ^        |            |
      SynthCoder |        |            |
                 |        |            |
              (f0,Ld)     | DDSP       | DDSP
                 ^        | Synth      | Synth
           CREPE |        |            |
                 |        |            v
               Audio      +------->  Audio
               Input             Reconstruction

  The MidiAutoencoder has two "branches":
    - The Inner Branch or "SynthCoder" Branch. This is the path that
      goes from Audio Input, CREPE, SynthCoder, DDSP Synth, Audio
      Reconstruction in the diagram above.
      This branch has the same set up
      as the DDSP Autoencoder described in the original DDSP ICLR 2020 paper. It
      learns a parameterized encoding of the input audio (i.e., a harmonic
      disribution, noise coefficients, etc).

    - The Outer Branch. This path starts at the output of the SynthCoder, goes
      to the MidiEncoder, the MidiDecoder, to the DDSP Synth and to a (separate)
      Audio Reconstruction.
      This branch uses the parameterized audio output of
      the SynthCoder Branch as input to produce a MIDI-like representation of
      the audio. This is another autoencoder network with the MIDI at its center
      and outputs another set of parameterized audio features to reconstruct the
      input audio. The parts of this second autoencoder are referred to as the
      MIDI encoder and MIDI decoder, respectively.

  In essence, each of these branches is its own Autoencoder and they're stacked
  on top of each other such that the output of the Inner Branch is fed as
  input to the Outer Branch.

  This class defines a standard MidiAutoencoder w/ no bells and whistles. It
  uses a quantized 1D representation at the bottleneck to represent MIDI pitch.
  Further child classes (below) enhance this common functionality. This class
  acts as a base class for the other MidiAutoencoder variants.
  """

  def __init__(self,
               synthcoder=None,
               midi_encoder=None,
               midi_decoder=None,
               sg_before_midiae=True,
               reverb=True,
               preprocessor=None,
               reconstruction_losses=None,
               qpitch_f0rec_loss=None,
               pitch_f0rec_loss=None,
               pitch_qpitch_loss=None,
               midi_slowness_loss=None,
               mask_f0_loss=True,
               **kwargs):
    """Constructor.

    Args:
      synthcoder: Callable network that decodes f0, loudness, z to synth
        parameters (amplitude, noise magnitudes, and a harmonic distribution).
      midi_encoder: Callable network that will be used to create a MIDI-like
        latent.
      midi_decoder: Callable network that will turn the MIDI latent vector back
        into f0 and amplitude features.
      sg_before_midiae: Bool indicating whether a stop gradient will happen
        before the MIDI space autoencoder.
      reverb: Bool controlling whether reverb is added to the processor group.
      preprocessor: Module to preprocess the input batch.
      reconstruction_losses: LossGroup for any and all things that need to be
        reconstructed (synth_audio, f0, amps, hd, noise, f0, etc).
      qpitch_f0rec_loss: Margin loss obj between quantized pitch and f0
        reconstruction. Ignored if `None`.
      pitch_f0rec_loss: Margin loss obj between unquantized pitch and f0
        reconstruction. Ignored if `None`.
      pitch_qpitch_loss: L-norm loss obj between unquantized pitch and quantized
        pitch. Ignored if `None`.
      midi_slowness_loss: Callable loss object that computes how slow the latent
        space is. Ignored if `None`.
      mask_f0_loss: Ignore f0 reconstruction loss where f0 and MIDI are very
        different.
      **kwargs: Other kwargs for keras models including `name`.
    """
    super().__init__(**kwargs)

    # Network objects
    self.synthcoder = synthcoder
    self.midi_encoder = midi_encoder
    self.midi_decoder = midi_decoder

    # Reconstruction Loss functions
    self.reconstruction_losses = reconstruction_losses
    self.mask_f0_loss = mask_f0_loss

    # Closeness and slowness losses
    self.qpitch_f0rec_loss = qpitch_f0rec_loss
    self.pitch_f0rec_loss = pitch_f0rec_loss
    self.pitch_qpitch_loss = pitch_qpitch_loss
    self.slowness_loss = midi_slowness_loss

    # Options for training
    self.sg_before_midiae = sg_before_midiae

    # Pre & post processing
    self.preprocessor = preprocessor
    if isinstance(self.preprocessor, preprocessing.F0PowerPreprocessor):
      self.db_key = 'power_db'
    else:
      self.db_key = 'loudness_db'

    # Explicitly define the DAG & Processor Group
    harmonic_synth = ddsp.synths.Harmonic()
    noise = ddsp.synths.FilteredNoise()
    add = ddsp.processors.Add()

    dag = [
        (harmonic_synth, ['amplitudes', 'harmonic_distribution', 'f0_hz']),
        (noise, ['magnitudes']),
        (add, ['filtered_noise/signal', 'harmonic/signal']),
    ]

    if reverb:
      reverb_module = ddsp.effects.FilteredNoiseReverb(
          trainable=True,
          reverb_length=24000,
          n_frames=500,
          n_filter_banks=32,
          name='reverb')
      dag.append(((reverb_module), ['add/signal']))

    self.processor_group = ddsp.processors.ProcessorGroup(dag=dag)

  def encode_to_midi(self, *args):
    """Encodes *args (f0, amps, hd, noise) into MIDI with a network."""
    if self.sg_before_midiae:
      args = [tf.stop_gradient(arg) for arg in args]

    z_pitch, z_vel = self.midi_encoder(*args).values()
    q_pitch = nn.straight_through_int_quantization(z_pitch)

    z_vel *= 0.0
    q_vel = z_vel

    return z_pitch, q_pitch, z_vel, q_vel

  def synthesize_audio(self,
                       f0_hz,
                       db,
                       z=None,
                       training=False,
                       return_params=False):
    """Run synthcoder and processor group."""
    features = {'f0_hz': f0_hz, self.db_key: db}
    if z is not None:
      features['z'] = z

    features.update(self.preprocessor(features))
    features.update(self.synthcoder(features, training=training))
    synth_params = self.processor_group.get_controls(features)
    audio = self.processor_group.get_signal(synth_params)

    if return_params:
      return audio, synth_params
    else:
      return audio

  @staticmethod
  def extract_harm_controls(synth_params, log_scale=True, stop_gradient=False):
    """Get harmonic synth controls from the outputs of the processor group."""
    amps = synth_params['harmonic']['controls']['amplitudes']
    hd = synth_params['harmonic']['controls']['harmonic_distribution']
    noise = synth_params['filtered_noise']['controls']['magnitudes']

    if log_scale:
      amps = ddsp.core.amplitude_to_db(amps, use_tf=True)
      noise = ddsp.core.amplitude_to_db(noise, use_tf=True)

    if stop_gradient:
      amps = tf.stop_gradient(amps)
      hd = tf.stop_gradient(hd)
      noise = tf.stop_gradient(noise)

    return amps, hd, noise

  @staticmethod
  def pianoroll_to_midi(pianoroll):
    """Converts a piano roll into conditioing format for the decoder."""

    # HACK(emanilow): This will only work for single-note melodies
    notes = tf.argmax(pianoroll, axis=-1)
    notes = tf.cast(notes, tf.float32)[:, :, tf.newaxis]
    velocities = tf.reduce_max(pianoroll, axis=-1)[:, :, tf.newaxis]
    return notes, velocities

  @staticmethod
  def midi_to_pianoroll(q_pitch, q_vel, piano_keys=128, thresh=20.0):
    """Converts quantized, encoded MIDI space representation to piano roll."""
    # pylint: disable=unused-argument
    # HACK(emanilow): This will only work for single-note melodies
    # HACK(emanilow): Ignores velocity!
    midi_space = tf.squeeze(q_pitch)
    if thresh > 0.0:
      midi_space = midi_space * tf.cast(midi_space > thresh, midi_space.dtype)
    return tf.one_hot(tf.cast(midi_space, tf.int32), piano_keys)

  def add_closeness_loss(self, loss_obj, f0, pitch, tag=''):
    """Applies loss_obj on abs(f0 - pitch)."""
    # pylint: disable=unused-argument
    closeness = tf.abs(pitch - f0)
    self._update_losses_dict(loss_obj, closeness)

  def add_slowness_loss(self, loss_obj, z_pitch, q_pitch, tag=''):
    """Applies slowness loss to pitch encoding for short notes."""
    if loss_obj is not None:
      note_mask = nn.get_note_mask(q_pitch, note_on_only=False)
      note_lengths = nn.get_note_lengths(note_mask)
      note_pitches = nn.get_note_moments(q_pitch, note_mask, return_std=False)
      loss_mask = nn.get_short_note_loss_mask(note_mask,
                                              note_lengths[..., 0],
                                              note_pitches[..., 0],
                                              min_length=40)
      loss_name = f'{loss_obj.name}{tag}'
      self._losses_dict[loss_name] = self.slowness_loss(z_pitch, loss_mask)

  def add_zpitch_losses(self, z_pitch, q_pitch, f0_midi_pred):
    """Helper function to update pitch encoder losses (if z_pitch available)."""
    if z_pitch is not None:
      self.add_slowness_loss(self.slowness_loss, z_pitch, q_pitch)
      self._update_losses_dict(self.pitch_qpitch_loss, z_pitch, q_pitch)
      self.add_closeness_loss(self.pitch_f0rec_loss, f0_midi_pred, z_pitch)

  def midi_to_audio(self, q_pitch, q_vel, z=None, return_synth_params=False):
    """Decode midi or piano roll to audio."""
    pg_in = self.midi_decoder(q_pitch, q_vel, z)
    midi_synth_params = self.processor_group.get_controls(pg_in)
    midi_audio = self.processor_group.get_signal(midi_synth_params)

    if return_synth_params:
      return midi_audio, midi_synth_params
    else:
      return midi_audio

  def get_audio_from_outputs(self, outputs):
    """Extract audio output tensor from outputs dict of call()."""
    audio_out = (outputs['midi_audio'] if self.midi_decoder is None else
                 outputs['synth_audio'])
    return audio_out

  def preprocess(self, features):
    """Run preprocessor and add extra keys to `features`."""

    features.update(self.preprocessor(features))
    features['f0_midi'] = ddsp.core.hz_to_midi(features['f0_hz'])
    features['db'] = features[self.db_key]

    return features

  def synthcoder_branch(self, features, training=True, z=None):
    """Wrapper to run the synthcoder branch."""
    f0_hz = features['f0_hz']
    db = features['db']
    synth_audio, synth_params = self.synthesize_audio(
        f0_hz, db, z=z, training=training, return_params=True)

    return synth_audio, synth_params

  def get_gt_midi(self, features):
    """Prepare the ground truth MIDI."""
    pr = features['note_active_velocities']
    f0_midi = features['f0_midi']
    q_pitch, q_vel = self.pianoroll_to_midi(pr)

    q_vel *= 0.0  # (Jesse) Don't use velocities for now.

    f0_loss_weights = None
    if self.mask_f0_loss:
      f0_loss_weights = tf.cast(tf.abs(f0_midi - q_pitch) < 2.0, tf.float32)

    return q_pitch, q_vel, f0_loss_weights

  def call(self, features, training=True):
    """Run the network to get a prediction and optionally get losses."""

    # --- Set up
    features = self.preprocess(features)

    # --- Synthcoder Branch
    synth_audio, synth_params = self.synthcoder_branch(features, training)
    amps, hd, noise = self.extract_harm_controls(synth_params)

    # --- MIDI Encoding
    z_pitch = None
    if self.midi_encoder is not None:
      # Encode with network.
      f0_midi = features['f0_midi']
      f0_loss_weights = tf.ones_like(f0_midi)
      z_pitch, q_pitch, _, q_vel = self.encode_to_midi(f0_midi, amps, hd, noise)
    else:
      # Skip encoder, use ground truth labels from MIDI.
      q_pitch, q_vel, f0_loss_weights = self.get_gt_midi(features)

    # --- MIDI Decoding
    if self.midi_decoder is None:
      f0_midi_pred = q_pitch  # HACK(jesseengel): Dummy values for summaries.
      midi_synth_params = synth_params
    else:

      # --- MIDI Decoding
      pg_in = self.midi_decoder(q_pitch, q_vel)
      f0_midi_pred = pg_in['f0_midi']
      midi_synth_params = self.processor_group.get_controls(pg_in)
      amps_pred, hd_pred, noise_pred = self.extract_harm_controls(
          midi_synth_params)

      # --- MIDI Audio Losses
      midi_audio = self.processor_group.get_signal(midi_synth_params)

      if training:
        # --- Pitch Encoder Losses
        self.add_zpitch_losses(z_pitch, q_pitch, f0_midi_pred)

        # --- Pitch Decoder Losses
        self.add_closeness_loss(self.qpitch_f0rec_loss, f0_midi_pred, q_pitch)

    # --- Finalize and return
    outputs = {
        'synth_params': synth_params,
        'synth_audio': synth_audio,
        'midi_synth_params': midi_synth_params,
        'midi_audio': midi_audio,
        'q_pitch': q_pitch,
        'q_vel': q_vel,
        'z_pitch': z_pitch,
        'pianoroll': self.midi_to_pianoroll(q_pitch, q_vel),
        'f0_midi_pred': f0_midi_pred,
        'f0_hz_pred': ddsp.core.midi_to_hz(f0_midi_pred),
        'amps': amps,
        'hd': hd,
        'noise': noise,
        'amps_pred': amps_pred,
        'hd_pred': hd_pred,
        'noise_pred': noise_pred,
        'f0_loss_weights': f0_loss_weights,
        f'{self.db_key}_pred': features['db']
    }
    # Remove unused variables (initialized as None)
    outputs = {k: v for k, v in outputs.items() if v is not None}

    outputs.update({k: v for k, v in features.items() if k not in outputs})
    outputs.update(synth_params)

    if training:
      loss_outs = self.reconstruction_losses(outputs)
      self._losses_dict.update(loss_outs)

    return outputs


class ZMidiAutoencoder(MidiAutoencoder):
  """A MidiAutoencoder that has additional z encoders.

  This is set up similar to the base MidiAutoencoder, except that there are
  many more places to learn encoding "z" variables to enable better
  reconstruction or better learnable representations.
  """

  def __init__(self,
               synthcoder=None,
               midi_encoder=None,
               midi_decoder=None,
               sg_before_midiae=True,
               reverb=True,
               preprocessor=None,
               reconstruction_losses=None,
               qpitch_f0rec_loss=None,
               pitch_f0rec_loss=None,
               pitch_qpitch_loss=None,
               midi_slowness_loss=None,
               mask_f0_loss=True,
               z_synth_encoders=None,
               z_global_encoders=None,
               z_note_encoder=None,
               z_preconditioning_stack=None,
               z_global_prior=None,
               z_note_prior=None,
               **kwargs):
    """Constructor.

    Args:
      synthcoder: Callable network that decodes f0, loudness, z to synth
        parameters (amplitude, noise magnitudes, and a harmonic distribution).
      midi_encoder: Callable network that will be used to create a MIDI-like
        latent.
      midi_decoder: Callable network that will turn the MIDI latent vector back
        into f0 and amplitude features.
      sg_before_midiae: Bool indicating whether a stop gradient will happen
        before the MIDI space autoencoder.
      reverb: Bool controlling whether reverb is added to the processor group.
      preprocessor: Module to preprocess the input batch.
      reconstruction_losses: LossGroup for any and all things that need to be
        reconstructed (synth_audio, f0, amps, hd, noise, f0, etc).
      qpitch_f0rec_loss: Margin loss obj between quantized pitch and f0
        reconstruction. Ignored if `None`.
      pitch_f0rec_loss: Margin loss obj between unquantized pitch and f0
        reconstruction. Ignored if `None`.
      pitch_qpitch_loss: L-norm loss obj between unquantized pitch and quantized
        pitch. Ignored if `None`.
      midi_slowness_loss: Callable loss object that computes how slow the latent
        space is. Ignored if `None`.
      mask_f0_loss: Ignore f0 reconstruction loss where f0 and MIDI are very
        different.
      z_synth_encoders: A list of callable networks that encodes audio as
        latents.
      z_global_encoders: A list of callable networks that will return non-MIDI
        latent representations.
      z_note_encoder: An encoder that encodes per-note latents, only for
        MIDI decoding, not for the synthcoder.
      z_preconditioning_stack: Optional callable network that will be used
        on the concatenated output of the z_global_encoders.
      z_global_prior: Prior loss on the global latents.
      z_note_prior: Prior loss on the note latents.
      **kwargs: Other kwargs for keras models including `name`.
    """
    super().__init__(synthcoder=synthcoder,
                     midi_encoder=midi_encoder,
                     midi_decoder=midi_decoder,
                     sg_before_midiae=sg_before_midiae,
                     reverb=reverb,
                     preprocessor=preprocessor,
                     reconstruction_losses=reconstruction_losses,
                     qpitch_f0rec_loss=qpitch_f0rec_loss,
                     pitch_f0rec_loss=pitch_f0rec_loss,
                     pitch_qpitch_loss=pitch_qpitch_loss,
                     midi_slowness_loss=midi_slowness_loss,
                     mask_f0_loss=mask_f0_loss,
                     **kwargs)

    self.z_synth_encoders = z_synth_encoders
    self.z_global_encoders = z_global_encoders
    self.z_note_encoder = z_note_encoder
    self.z_preconditioning_stack = z_preconditioning_stack
    self.z_global_prior = z_global_prior
    self.z_note_prior = z_note_prior

  def z_synth_encode(self, features):
    """Encode other latents besides pitch and velocity."""
    z = None
    if self.z_synth_encoders is not None:
      z = [z_encoder(features)['z'] for z_encoder in self.z_synth_encoders]
      z = tf.concat(z, axis=-1)
    return z

  def z_global_encode(self, features):
    """Encode other latents besides pitch and velocity."""
    z = None
    if self.z_global_encoders is not None:
      z = [z_encoder(features)['z'] for z_encoder in self.z_global_encoders]
      z = tf.concat(z, axis=-1)
    return z

  def z_note_encode(self, features, q_pitch):
    """Encode latents per a note with local pooling."""
    z_notes = None
    if self.z_note_encoder is not None:
      z_notes = self.z_note_encoder(features)['z']
      note_mask = nn.get_note_mask(q_pitch)
      z_notes = nn.pool_over_notes(z_notes, note_mask)
    return z_notes

  def call(self, features, training=True):
    """Run the network to get a prediction and optionally get losses."""

    # --- Set up
    features = self.preprocess(features)

    # --- Synthcoder Branch
    z_synth = self.z_synth_encode(features)
    synth_audio, synth_params = self.synthcoder_branch(features, training,
                                                       z_synth)
    amps, hd, noise = self.extract_harm_controls(synth_params)

    # Add to features for z_note_encoder.
    features['amps_scaled'] = amps
    features['hd_scaled'] = hd
    features['noise_scaled'] = noise

    # --- MIDI Encoding
    z_pitch, z_vel = None, None
    if self.midi_encoder is not None:
      # Encode with network.
      f0_midi = features['f0_midi']
      f0_loss_weights = tf.ones_like(f0_midi)
      z_pitch, q_pitch, _, q_vel = self.encode_to_midi(f0_midi, amps, hd, noise)

    else:
      # Skip encoder, use ground truth labels from MIDI.
      q_pitch, q_vel, f0_loss_weights = self.get_gt_midi(features)

    # --- MIDI Decoding
    if self.midi_decoder is None:
      f0_midi_pred = q_pitch  # HACK(jesseengel): Dummy values for summaries.
      midi_synth_params = synth_params
      z_global, z_notes = None, None
    else:
      # --- Z Global Encoding
      z_global = self.z_global_encode(features)

      # --- Z Note Encoding
      z_notes = self.z_note_encode(features, q_pitch)

      # Apply priors.
      if self.z_global_prior is not None:
        self._update_losses_dict(self.z_global_prior, z_global)
        z_global = self.z_global_prior(z_global)  # Sample posterior.

      if self.z_note_prior is not None:
        self._update_losses_dict(self.z_note_prior, z_notes)
        z_notes = self.z_note_prior(z_notes)  # Sample posterior.

      # Pack the Z together.
      if z_notes is None:
        z_midi_decoder = z_global
      else:
        z_midi_decoder = tf.concat([z_global, z_notes], axis=-1)

      # --- Precondition
      if self.z_preconditioning_stack is not None:
        z_midi_decoder = self.z_preconditioning_stack(z_midi_decoder)

      # --- MIDI Decoding
      pg_in = self.midi_decoder(q_pitch, q_vel, z_midi_decoder)
      f0_midi_pred = pg_in['f0_midi']
      midi_synth_params = self.processor_group.get_controls(pg_in)
      amps_pred, hd_pred, noise_pred = self.extract_harm_controls(
          midi_synth_params)

      # --- MIDI Audio Losses
      midi_audio = self.processor_group.get_signal(midi_synth_params)
      if training:
        # --- Pitch Decoder Losses
        self.add_closeness_loss(self.qpitch_f0rec_loss, f0_midi_pred, q_pitch)

        # --- Pitch Encoder Losses
        self.add_zpitch_losses(z_pitch, q_pitch, f0_midi_pred)

    # --- Finalize and return
    outputs = {
        'synth_params': synth_params,
        'synth_audio': synth_audio,
        'midi_synth_params': midi_synth_params,
        'midi_audio': midi_audio,
        'q_pitch': q_pitch,
        'q_vel': q_vel,
        'z_pitch': z_pitch,
        'z_vel': z_vel,
        'z_global': z_global,
        'z_notes': z_notes,
        'pianoroll': self.midi_to_pianoroll(q_pitch, q_vel),
        'f0_midi_pred': f0_midi_pred,
        'f0_hz_pred': ddsp.core.midi_to_hz(f0_midi_pred),
        'amps': amps,
        'hd': hd,
        'noise': noise,
        'amps_pred': amps_pred,
        'hd_pred': hd_pred,
        'noise_pred': noise_pred,
        'f0_loss_weights': f0_loss_weights,
        f'{self.db_key}_pred': features['db']
    }
    # Remove unused z vectors (initialized as None)
    outputs = {k: v for k, v in outputs.items() if v is not None}

    outputs.update({k: v for k, v in features.items() if k not in outputs})
    outputs.update(synth_params)

    if training:
      loss_outs = self.reconstruction_losses(outputs)
      self._losses_dict.update(loss_outs)

    return outputs


