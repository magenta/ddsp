
import tensorflow as tf
import ddsp
import ddsp.training

from utils.inference_utils import get_process_group
from .interpretable_conditioning import extract_harm_controls


class SynthCoder(tf.keras.Model):
  def __init__(self, encoder, decoder):
    super(SynthCoder, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.model_type = 'SynthCoder'

  def encode(self, inputs, training=False):
    z = self.encoder(inputs, training=training)
    return z

  def decode(self, z, inputs):
    synth_params = self.decoder([z, inputs])
    return synth_params

  def call(self, inputs, training=None, reverb=False):
    z = self.encode(inputs, training=training)
    synth_params = self.decode(z, inputs)
    return synth_params

  def _build(self, inputs):
    inputs, kwargs = inputs
    self(inputs, **kwargs)


class MIDIAEInterpCond(tf.keras.Model):
  """
  TODO: draw a diagram on https://asciiflow.com/#/, like below:
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
  """

  def __init__(self, synth_coder, midi_decoder, n_frames=1000, frame_size=64,
               sample_rate=16000, reverb_module=None,
               use_f0_ld=False):
    super(MIDIAEInterpCond, self).__init__()
    self.synth_coder = synth_coder
    self.midi_decoder = midi_decoder
    self.reverb_module = reverb_module
    self.model_type = 'MIDIAEInterpCond'

    self.processor_group = get_process_group(n_frames, frame_size, sample_rate,
                                             use_angular_cumsum=False)
    self.n_frames = n_frames
    self.frame_size = frame_size
    self.sample_rate = sample_rate

    self.use_f0_ld = use_f0_ld

  def train_synth_coder_only(self):
    self.midi_decoder.trainable = False
    self.synth_coder.trainable = True

  def freeze_synth_coder(self):
    self.midi_decoder.trainable = True
    self.synth_coder.trainable = False
    if self.reverb_module is not None:
      self.reverb_module.trainable = False

  def run_synth_coder(self, features, training=True):
    synth_params = self.synth_coder(features, training=training)
    control_params = self.processor_group.get_controls(synth_params,
                                                       verbose=False)
    synth_audio = self.processor_group.get_signal(control_params)
    if self.reverb_module is not None:
      synth_audio = self.reverb_module(synth_audio,
                                       reverb_number=features['instrument_id'],
                                       training=training)
    return synth_params, control_params, synth_audio

  @staticmethod
  def get_gt_midi(features):
    """Prepare the ground truth MIDI."""
    f0_midi = ddsp.core.hz_to_midi(features['f0_hz'])
    q_pitch = tf.cast(features['midi'][..., tf.newaxis], tf.float32)

    q_vel = q_pitch * 0.0  # (Jesse) Don't use velocities for now.

    f0_loss_weights = tf.cast(tf.abs(f0_midi - q_pitch) < 2.0, tf.float32)

    return q_pitch, q_vel, f0_loss_weights, features['onsets'], features[
      'offsets']

  def gen_cond_dict_from_feature(self, features, training=False):
    synth_params, control_params, synth_audio = self.run_synth_coder(features,
                                                                     training=training)

    # synth_params_normalized: scaled and normalized synth params
    synth_params_normalized = extract_harm_controls(control_params)

    midi_features = self.get_gt_midi(features)

    conditioning_dict = self.midi_decoder.gen_cond_dict(synth_params_normalized,
                                                        midi_features)

    return synth_params_normalized, midi_features, conditioning_dict

  def gen_audio_from_cond_dict(self, conditioning_dict, midi_features,
                               instrument_id=None,
                               use_angular_cumsum=True,
                               display_progressbar=False):
    z_midi_decoder, params_pred = self.midi_decoder.gen_params_from_cond(
      conditioning_dict, midi_features,
      instrument_id=instrument_id,
      display_progressbar=display_progressbar)

    self.processor_group = get_process_group(z_midi_decoder.shape[1],
                                             self.frame_size, self.sample_rate,
                                             use_angular_cumsum=use_angular_cumsum)

    if self.use_f0_ld:
      midi_synth_params, midi_control_params, midi_audio = self.run_synth_coder(
        {'f0_hz': params_pred['f0_hz'],
         'loudness_db': params_pred['ld'],
         'instrument_id': instrument_id},
        training=False)
      f0_pred, amps_pred, hd_pred, noise_pred = extract_harm_controls(
        midi_control_params)

    else:
      midi_synth_params = self.processor_group.get_controls(
        {'amplitudes': params_pred['amplitudes'],
         'harmonic_distribution': params_pred[
           'harmonic_distribution'],
         'noise_magnitudes': params_pred['noise_magnitudes'],
         'f0_hz': params_pred['f0_hz'], },
        verbose=False)

      midi_audio = self.processor_group.get_signal(midi_synth_params)
      if self.reverb_module is not None:
        midi_audio = self.reverb_module(midi_audio, reverb_number=instrument_id,
                                        training=False)
      f0_pred, amps_pred, hd_pred, noise_pred = extract_harm_controls(
        midi_synth_params)

    midi_control_params = (f0_pred, amps_pred, hd_pred, noise_pred)
    return midi_audio, midi_control_params, midi_synth_params

  def call(self, features, training=False, run_synth_coder_only=False):
    """Run the network to get a prediction and optionally get losses."""

    synth_params, control_params, synth_audio = self.run_synth_coder(features,
                                                                     training=training)
    if run_synth_coder_only:  # only run synth coder branch
      outputs = {
        'synth_params': synth_params,
        'synth_audio': synth_audio,
      }
      return outputs

    # synth_params_normalized: scaled and normalized synth params
    synth_params_normalized = extract_harm_controls(control_params,
                                                    stop_gradient=True)

    midi_features = self.get_gt_midi(features)

    # --- MIDI Decoding
    conditioning_dict, params_pred = self.midi_decoder(features,
                                                       synth_params_normalized,
                                                       midi_features,
                                                       training=training,
                                                       synth_params=control_params)

    if self.use_f0_ld:
      midi_synth_params, midi_control_params, midi_audio = self.run_synth_coder(
        {'f0_hz': params_pred['f0_hz'],
         'loudness_db': params_pred['ld'],
         'instrument_id': features[
           'instrument_id']
         },
        training=training)
      f0_pred, amps_pred, hd_pred, noise_pred = extract_harm_controls(
        midi_control_params)

    else:
      midi_synth_params = self.processor_group.get_controls(
        {'amplitudes': params_pred['amplitudes'],
         'harmonic_distribution': params_pred[
           'harmonic_distribution'],
         'noise_magnitudes': params_pred['noise_magnitudes'],
         'f0_hz': params_pred['f0_hz'], },
        verbose=False)
      # --- MIDI Audio Losses
      midi_audio = self.processor_group.get_signal(midi_synth_params)

      if self.reverb_module is not None:
        midi_audio = self.reverb_module(midi_audio,
                                        reverb_number=features['instrument_id'],
                                        training=training)

      f0_pred, amps_pred, hd_pred, noise_pred = extract_harm_controls(
        midi_synth_params)

    # unpack things
    f0, amps, hd, noise = synth_params_normalized
    q_pitch, q_vel, f0_loss_weights, onsets, offsets = midi_features

    # --- Finalize and return
    outputs = {
      'synth_params': synth_params,
      'control_params': control_params,
      'synth_audio': synth_audio,
      'midi_synth_params': midi_synth_params,
      'midi_audio': midi_audio,
      'q_pitch': q_pitch,
      'q_vel': q_vel,
      'conditioning_dict': conditioning_dict,
      'amps': amps,
      'hd': hd,
      'noise': noise,
      'amps_pred': amps_pred,
      'hd_pred': hd_pred,
      'noise_pred': noise_pred,
      'f0_loss_weights': f0_loss_weights,
      'params_pred': params_pred,
    }

    return outputs

  def _build(self, inputs):
    inputs, kwargs = inputs
    self(inputs, **kwargs)
