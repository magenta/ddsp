"""LossHelper for handling the reconstruction used in training."""

import tensorflow as tf
import tensorflow.keras.layers as layers
import ddsp
from midi_ddsp.loss_helper import LossHelper
from ddsp.losses import ParamLoss


class EvalReconLossGroup(layers.Layer):
  """Evaluation loss group, for calculating L1 and L2 loss
  on synthesis parameters and spectral loss on audio.
  It is used in evaluation phase regardless of the training objective."""

  def __init__(self):
    super(EvalReconLossGroup, self).__init__()
    self.f0_loss = ParamLoss(weight=50.0, loss_type='L2',
                             name='f0_reconstruction')
    self.amps_loss = ParamLoss(weight=0.5, loss_type='L1',
                               name='amplitude_reconstruction')
    self.hd_loss = ParamLoss(weight=500.0, loss_type='L1',
                             name='harmonic_distribution_reconstruction')
    self.noise_loss = ParamLoss(weight=0.5, loss_type='L1',
                                name='noise_reconstruction')
    self.synth_spectral_loss = ddsp.losses.SpectralLoss(loss_type='L1',
                                                        mag_weight=1.0,
                                                        logmag_weight=1.0)
    self.midi_spectral_loss = ddsp.losses.SpectralLoss(loss_type='L1',
                                                       mag_weight=1.0,
                                                       logmag_weight=1.0)

  def call(self, inputs, outputs, synth_coder_only=False, add_synth_loss=False):
    loss_spectral_synth = self.synth_spectral_loss(inputs['audio'],
                                                   outputs['synth_audio'])
    if synth_coder_only:
      loss_dict = {
        'loss_spectral_synth': loss_spectral_synth,
        'total_loss': loss_spectral_synth
      }
      return loss_dict

    loss_f0 = self.f0_loss(outputs['params_pred']['f0_midi'],
                           ddsp.core.hz_to_midi(inputs['f0_hz']),
                           weights=outputs['f0_loss_weights'])
    loss_amp = self.amps_loss(outputs['amps_pred'], outputs['amps'])
    loss_hd = self.hd_loss(outputs['hd_pred'], outputs['hd'])
    loss_noise = self.noise_loss(outputs['noise_pred'], outputs['noise'])
    loss_spectral_midi = self.midi_spectral_loss(inputs['audio'],
                                                 outputs['midi_audio'])
    loss_dict = {
      'loss_amp': loss_amp,
      'loss_f0': loss_f0,
      'loss_hd': loss_hd,
      'loss_noise': loss_noise,
      'loss_spectral_synth': loss_spectral_synth,
      'loss_spectral_midi': loss_spectral_midi,
      'total_loss': loss_f0 + loss_amp + loss_hd + loss_noise
    }

    return loss_dict


class ReconLossGroup(layers.Layer):
  """The reconstruction loss group used in MIDI-DDSP.
  It calculates back-prop (in total_loss) consists of spectral loss on audio,
  L1 and L2 loss on amplitude, harmonic distribution and noise magnitude, and
  cross-entropy loss on f0.
  However, other losses are also calculated but not back-proped to monitor
  the training process."""

  def __init__(self, midi_audio_loss=False, synth_params_loss=True):
    super(ReconLossGroup, self).__init__()
    self.amps_loss = ParamLoss(weight=0.5, loss_type='L1',
                               name='amplitude_reconstruction')
    self.hd_loss = ParamLoss(weight=500.0, loss_type='L1',
                             name='harmonic_distribution_reconstruction')
    self.noise_loss = ParamLoss(weight=0.5, loss_type='L1',
                                name='noise_reconstruction')
    self.synth_spectral_loss = ddsp.losses.SpectralLoss(loss_type='L1',
                                                        mag_weight=1.0,
                                                        logmag_weight=1.0)
    self.midi_audio_loss = midi_audio_loss
    self.synth_params_loss = synth_params_loss
    self.midi_spectral_loss = ddsp.losses.SpectralLoss(loss_type='L1',
                                                       mag_weight=1.0,
                                                       logmag_weight=1.0)

  def call(self, inputs, outputs, synth_coder_only=False, add_synth_loss=False):
    loss_spectral_synth = self.synth_spectral_loss(inputs['audio'],
                                                   outputs['synth_audio'])
    if synth_coder_only:
      loss_dict = {
        'loss_spectral_synth': loss_spectral_synth,
        'total_loss': loss_spectral_synth
      }
      return loss_dict

    f0_dv = ddsp.core.hz_to_midi(inputs['f0_hz']) - tf.stop_gradient(
      outputs['q_pitch'])
    f0_midi_bined = tf.cast(
      (tf.clip_by_value(f0_dv[:, :, -1], -1, 1) + 1) * 100, tf.int64)
    loss_f0 = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(f0_midi_bined,
                                                     outputs['params_pred'][
                                                       'prenet_f0_output'][
                                                       'f0_midi_dv_logits']))

    loss_amp = self.amps_loss(outputs['amps_pred'], outputs['amps'])
    loss_hd = self.hd_loss(outputs['hd_pred'], outputs['hd'])
    loss_noise = self.noise_loss(outputs['noise_pred'], outputs['noise'])

    loss_spectral_midi = self.midi_spectral_loss(inputs['audio'],
                                                 outputs['midi_audio'])
    loss_dict = {
      'loss_amp': loss_amp,
      'loss_f0': loss_f0,
      'loss_hd': loss_hd,
      'loss_noise': loss_noise,
      'loss_spectral_synth': loss_spectral_synth,
      'loss_spectral_midi': loss_spectral_midi,
    }
    if self.synth_params_loss:
      loss_dict['total_loss'] = loss_amp + loss_f0 + loss_hd + loss_noise
    else:
      loss_dict['total_loss'] = loss_f0
    if add_synth_loss:
      loss_dict['total_loss'] = loss_dict['total_loss'] + loss_dict[
        'loss_spectral_synth']
    if self.midi_audio_loss:
      loss_dict['total_loss'] = loss_dict['total_loss'] + loss_dict[
        'loss_spectral_midi']

    return loss_dict


class F0LdReconLossGroup(layers.Layer):
  """Reconstruction loss group for MIDI2Params.
  Basically calculate cross-entropy loss on f0 and loundess logits.
  However, L1 and L2 loss on synthesis parameters and spectral loss on audio
  are also calculated but not back-proped to monitor the training process. """

  def __init__(self, midi_audio_loss=False, synth_params_loss=True,
               full_f0_loss=False):
    super(F0LdReconLossGroup, self).__init__()
    self.hd_loss = ParamLoss(weight=500.0, loss_type='L1',
                             name='harmonic_distribution_reconstruction')
    self.noise_loss = ParamLoss(weight=0.5, loss_type='L1',
                                name='noise_reconstruction')
    self.synth_spectral_loss = ddsp.losses.SpectralLoss(loss_type='L1',
                                                        mag_weight=1.0,
                                                        logmag_weight=1.0)
    self.midi_audio_loss = midi_audio_loss
    self.synth_params_loss = synth_params_loss
    self.midi_spectral_loss = ddsp.losses.SpectralLoss(loss_type='L1',
                                                       mag_weight=1.0,
                                                       logmag_weight=1.0)
    self.full_f0_loss = full_f0_loss

  def call(self, inputs, outputs, synth_coder_only=False, add_synth_loss=False):
    loss_spectral_synth = self.synth_spectral_loss(inputs['audio'],
                                                   outputs['synth_audio'])
    if synth_coder_only:
      loss_dict = {
        'loss_spectral_synth': loss_spectral_synth,
        'total_loss': loss_spectral_synth
      }
      return loss_dict

    amps_bined = tf.cast(
      tf.clip_by_value(tf.stop_gradient(inputs['loudness_db'] + 120)[:, :, -1],
                       0, 120),
      tf.int64)
    if self.full_f0_loss:
      f0_midi = ddsp.core.hz_to_midi(inputs['f0_hz'])
      f0_midi_bined = tf.cast(
        (tf.clip_by_value(f0_midi[:, :, -1], 24, 96) - 24) * 5, tf.int64)
      loss_f0 = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(f0_midi_bined,
                                                       outputs['params_pred'][
                                                         'f0_midi_logits']))

    else:
      f0_dv = ddsp.core.hz_to_midi(inputs['f0_hz']) - tf.stop_gradient(
        outputs['q_pitch'])
      f0_midi_bined = tf.cast(
        (tf.clip_by_value(f0_dv[:, :, -1], -1, 1) + 1) * 100, tf.int64)
      loss_f0 = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(f0_midi_bined,
                                                       outputs['params_pred'][
                                                         'f0_midi_dv_logits']))
    loss_amp = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(amps_bined,
                                                     outputs['params_pred'][
                                                       'ld_logits']))
    loss_hd = self.hd_loss(outputs['hd_pred'], outputs['hd'])
    loss_noise = self.noise_loss(outputs['noise_pred'], outputs['noise'])
    loss_spectral_midi = self.midi_spectral_loss(inputs['audio'],
                                                 outputs['midi_audio'])
    loss_dict = {
      'loss_amp': loss_amp,
      'loss_f0': loss_f0,
      'loss_hd': loss_hd,
      'loss_noise': loss_noise,
      'loss_spectral_synth': loss_spectral_synth,
      'loss_spectral_midi': loss_spectral_midi,
      'total_loss': loss_amp + loss_f0,
    }
    if add_synth_loss:
      loss_dict['total_loss'] = loss_dict['total_loss'] + loss_dict[
        'loss_spectral_synth']

    return loss_dict


class ReconLossHelper(LossHelper):
  """Overall wrapper for getting the reconstruction loss helper
  based on hyperparameters."""

  def __init__(self, hp, eval_recon_loss=False):
    midi_audio_loss = hp.midi_audio_loss
    synth_params_loss = hp.synth_params_loss
    midi_decoder_type = hp.midi_decoder_type
    midi_decoder_decoder_net = hp.midi_decoder_decoder_net
    super(ReconLossHelper, self).__init__()

    self.loss_list = [
      'loss_amp',
      'loss_f0',
      'loss_hd',
      'loss_noise',
      'loss_spectral_synth',
      'loss_spectral_midi',
      'total_loss'
    ]

    if eval_recon_loss:
      self.loss_group = EvalReconLossGroup()

    elif midi_decoder_type == 'midi_decoder' or \
          midi_decoder_type == 'unconditioned' or \
          midi_decoder_decoder_net == 'rnn_f0_ld':

      self.loss_group = F0LdReconLossGroup(midi_audio_loss=midi_audio_loss,
                                           synth_params_loss=synth_params_loss,
                                           full_f0_loss=
                                           midi_decoder_type == 'unconditioned')
    elif midi_decoder_type == 'interpretable_conditioning':
      if 'dilated_conv' in midi_decoder_decoder_net:
        self.loss_group = EvalReconLossGroup()
      else:
        self.loss_group = ReconLossGroup(midi_audio_loss=midi_audio_loss,
                                         synth_params_loss=synth_params_loss)

    self.init_metrics()
