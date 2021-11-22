"""Loss helper for handling and logging GAN loss."""
import tensorflow as tf
from midi_ddsp.loss_helper import LossHelper
import ddsp
from ddsp.spectral_ops import F0_RANGE


class GANLossHelper(LossHelper):
  def __init__(self, lambda_recon=1, lambda_feat=10, lambda_G=1, sg_z=True):
    """Loss helper for handling and logging GAN loss.

    Args:
      lambda_recon: the reconstruction coefficient.
      lambda_feat: the feature matching loss coefficient.
      lambda_G: the generator loss coefficient.
      sg_z: whether to stop gradient on the feature sequence input to the
        synthesis generator.
    """
    super(GANLossHelper, self).__init__()
    self.loss_list = [
      'disc_loss',
      'gen_loss',
      'fm_loss'
    ]
    self.n_layers_D = 4
    self.num_D = 3
    self.lambda_feat = lambda_feat
    self.lambda_recon = lambda_recon
    self.lambda_G = lambda_G
    self.sg_z = sg_z
    self.init_metrics()

  def compute_disc_loss(self, D_fake_det, D_real):
    loss_D = 0
    for scale in D_fake_det:
      loss_D += tf.reduce_mean(scale[-1] ** 2)

    for scale in D_real:
      loss_D += tf.reduce_mean((scale[-1] - 1) ** 2)

    return {'disc_loss': loss_D}

  def compute_gen_loss(self, D_fake, D_real, total_loss_recon):
    loss_G = 0
    for scale in D_fake:
      loss_G += tf.reduce_mean((scale[-1] - 1) ** 2)

    loss_feat = 0
    feat_weights = 4.0 / (self.n_layers_D + 1)
    D_weights = 1.0 / self.num_D
    wt = D_weights * feat_weights
    for i in range(self.num_D):
      for j in range(len(D_fake[i]) - 1):
        loss_feat += wt * tf.reduce_mean(
          tf.abs(D_fake[i][j] - tf.stop_gradient(D_real[i][j])))  # MAE

    fm_loss = self.lambda_feat * loss_feat

    total_loss = self.lambda_recon * total_loss_recon + \
                 self.lambda_G * loss_G + fm_loss

    return total_loss, {'gen_loss': loss_G, 'fm_loss': fm_loss}

  def get_disc_input(self, outputs):
    if self.sg_z:
      cond = tf.stop_gradient(outputs['params_pred']['z_midi_decoder'])
    else:
      cond = outputs['params_pred']['z_midi_decoder']

    real_outputs = tf.concat([
      ddsp.core.hz_to_midi(outputs['synth_params']['f0_hz'] / F0_RANGE),
      outputs['synth_params']['amplitudes'],
      outputs['synth_params']['harmonic_distribution'],
      outputs['synth_params']['noise_magnitudes']], axis=-1)
    real_outputs = tf.stop_gradient(real_outputs)

    fake_outputs = tf.concat([
      ddsp.core.hz_to_midi(outputs['params_pred']['f0_hz'] / F0_RANGE),
      outputs['params_pred']['amplitudes'],
      outputs['params_pred']['harmonic_distribution'],
      outputs['params_pred']['noise_magnitudes']], axis=-1)

    return cond, real_outputs, fake_outputs
