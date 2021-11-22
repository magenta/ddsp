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
"""Model base class."""

import time

from absl import logging
import ddsp
from ddsp.core import copy_if_tf_function
from ddsp.training import train_util
import tensorflow as tf


class Model(tf.keras.Model):
  """Base class for all models."""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._losses_dict = {}

  def __call__(self, *args, return_losses=False, **kwargs):
    """Reset the losses dict on each call.

    Args:
      *args: Arguments passed on to call().
      return_losses: Return a dictionary of losses in addition to the call()
        function returns.
      **kwargs: Keyword arguments passed on to call().

    Returns:
      outputs: A dictionary of model outputs generated in call().
        {output_name: output_tensor or dict}.
      losses: If return_losses=True, also returns a dictionary of losses,
        {loss_name: loss_value}.
    """
    # Copy mutable dicts if in graph mode to prevent side-effects (pure func).
    args = [copy_if_tf_function(a) if isinstance(a, dict) else a for a in args]

    # Run model.
    self._losses_dict = {}
    outputs = super().__call__(*args, **kwargs)

    # Get total loss.
    if not return_losses:
      return outputs
    else:
      self._losses_dict['total_loss'] = self.sum_losses(self._losses_dict)
      return outputs, self._losses_dict

  def sum_losses(self, losses_dict):
    """Sum all the scalar losses in a dictionary."""
    return tf.reduce_sum(list(losses_dict.values()))

  def _update_losses_dict(self, loss_objs, *args, **kwargs):
    """Helper function to run loss objects on args and add to model losses."""
    for loss_obj in ddsp.core.make_iterable(loss_objs):
      if hasattr(loss_obj, 'get_losses_dict'):
        losses_dict = loss_obj.get_losses_dict(*args, **kwargs)
        self._losses_dict.update(losses_dict)

  def restore(self, checkpoint_path, verbose=True):
    """Restore model and optimizer from a checkpoint.

    Args:
      checkpoint_path: Path to checkpoint file or directory.
      verbose: Warn about missing variables.

    Raises:
      FileNotFoundError: If no checkpoint is found.
    """
    start_time = time.time()
    latest_checkpoint = train_util.get_latest_checkpoint(checkpoint_path)
    checkpoint = tf.train.Checkpoint(model=self)
    if verbose:
      checkpoint.restore(latest_checkpoint)
    else:
      checkpoint.restore(latest_checkpoint).expect_partial()
    logging.info('Loaded checkpoint %s', latest_checkpoint)
    logging.info('Loading model took %.1f seconds', time.time() - start_time)

  def get_audio_from_outputs(self, outputs):
    """Extract audio output tensor from outputs dict of call()."""
    raise NotImplementedError('Must implement `self.get_audio_from_outputs()`.')

  def call(self, *args, training=False, **kwargs):
    """Run the forward pass, add losses, and create a dictionary of outputs.

    This function must run the forward pass, add losses to self._losses_dict and
    return a dictionary of all the relevant output tensors.

    Args:
      *args: Args for forward pass.
      training: Required `training` kwarg passed in by keras.
      **kwargs: kwargs for forward pass.

    Returns:
      Dictionary of all relevant tensors.
    """
    raise NotImplementedError('Must implement a `self.call()` method.')


class GANModel(Model):
  """Base class for all models with adversarial losses that use GANTrainer.

  Must implement 2 properties and 2 methods:
    - discriminator_variables
    - generator_variables
    - get_discriminator_losses()
    - get_generator_losses()
  """

  def __call__(self, *args, return_losses=False, **kwargs):
    """Reset the losses dict on each call.

    Args:
      *args: Arguments passed on to call().
      return_losses: Return a dictionary of losses in addition to the call()
        function returns.
      **kwargs: Keyword arguments passed on to call().

    Returns:
      outputs: A dictionary of model outputs generated in call().
        {output_name: output_tensor or dict}.
      losses: If return_losses=True, also returns a dictionary of losses,
        {loss_name: loss_value}.
    """
    # Copy mutable dicts if in graph mode to prevent side-effects (pure func).
    args = [copy_if_tf_function(a) if isinstance(a, dict) else a for a in args]

    # Run model.
    self._losses_dict = {}
    outputs = super().__call__(*args, **kwargs)

    # Get total loss.
    if not return_losses:
      return outputs
    else:
      losses = self._losses_dict

      # Scale and separate losses for each component.
      g_losses = self.get_generator_losses(losses)
      d_losses = self.get_discriminator_losses(losses)
      other_losses = self.get_other_losses(losses)

      # Get total losses.
      total_g_loss = self.sum_losses(g_losses)
      total_d_loss = self.sum_losses(d_losses)
      total_other_loss = self.sum_losses(other_losses)

      losses['total_g_loss'] = total_g_loss
      losses['total_d_loss'] = total_d_loss
      losses['total_other_loss'] = total_other_loss

    losses['total_loss'] = total_g_loss + total_d_loss + total_other_loss
    self._losses_dict = losses

    return outputs, self._losses_dict

  @property
  def discriminator_variables(self):
    """Return all variables for discriminator networks."""
    raise NotImplementedError(
        'Must implement method to get discriminator variables.')

  @property
  def generator_variables(self):
    """Return all variables for generator networks."""
    raise NotImplementedError(
        'Must implement method to get generator variables.')

  @property
  def other_variables(self):
    """Return all non-GAN variables."""
    gan_variables = self._set_union(
        self.discriminator_variables, self.generator_variables)
    return self._set_difference(self.trainable_variables, gan_variables)

  def get_discriminator_losses(self, losses_dict):
    """Filter losses_dict and return losses for discriminator networks."""
    raise NotImplementedError(
        'Must implement method to get discriminator losses from losses_dict.')

  def get_generator_losses(self, losses_dict):
    """Filter losses_dict and return losses for generator networks."""
    raise NotImplementedError(
        'Must implement method to get generator losses from losses_dict.')

  def get_other_losses(self, losses_dict):
    """Filter losses_dict and return non GAN losses."""
    gan_keys = []
    gan_keys += list(self.get_discriminator_losses(losses_dict).keys())
    gan_keys += list(self.get_generator_losses(losses_dict).keys())
    return {k: v for k, v in losses_dict.items() if k not in gan_keys}

  @staticmethod
  def _set_difference(var_list_1, var_list_2):
    """Set-wise difference between two lists of variables."""
    as_refs = lambda l: [v.ref() for v in l]
    as_vars = lambda l: [v.deref() for v in l]
    return as_vars(set(as_refs(var_list_1)).difference(as_refs(var_list_2)))

  @staticmethod
  def _set_union(var_list_1, var_list_2):
    """Set-wise difference between two lists of variables."""
    as_refs = lambda l: [v.ref() for v in l]
    as_vars = lambda l: [v.deref() for v in l]
    return as_vars(set(as_refs(var_list_1)).union(as_refs(var_list_2)))


class MultiScaleVQGAN(GANModel):
  """Parallel architecture with Discriminator loss at each scale."""

  def __init__(self,
               wavelet_decomposition=None,
               encoders=None,
               decoders=None,
               quantizers=None,
               discriminators=None,
               audio_losses=None,
               band_losses=None,
               gan_losses=None,
               input_length=None,
               gan_loss_ratio=1.0,
               max_gan_loss=4.0,
               **kwargs):
    super().__init__(**kwargs)
    self.discriminators = self._check_n_bands(discriminators)
    self.gan_losses = self._check_n_bands(gan_losses)
    self.z_gan_dims = 128
    self.gan_loss_ratio = gan_loss_ratio
    self.max_gan_loss = max_gan_loss

    # Get GAN Loss keys.
    self.d_loss_keys = []
    for gan_loss in self.gan_losses:
      self.d_loss_keys += gan_loss.discriminator_loss_keys

    self.g_loss_keys = []
    for gan_loss in self.gan_losses:
      self.g_loss_keys += gan_loss.generator_loss_keys

  def call(self, features, training=True):
    # Initial user-specified trimming/padding.
    audio = self.trim_or_pad_audio(features['audio'])

    # Decompose.
    bands, audio_padded = self.wd.decompose(audio)

    # Encode.
    zs = self.encode(bands)

    # Quantize.
    z_qs, cs = self.quantize(zs)

    # Sample GAN latent.
    z_gan = tf.random.normal([audio.shape[0], 1, self.z_gan_dims])

    # Combine with quantized latents.
    z_decodes = [z_q + z_gan for z_q in z_qs]

    # Decode.
    bands_rec = self.decode(z_decodes)

    # Recompose.
    audio_rec, bands_rec_up = self.wd.recompose(bands_rec)

    # Discriminate.
    ds_real, ds_rec = self.discriminate(bands, bands_rec)

    print('!Padded Audio!', audio_padded.shape)
    print('!Z_Q!', [z.shape for z in z_qs])
    print('!Reconstruction Audio!', audio_rec.shape)
    print('!Bands!', [b.shape for b in bands])
    print('!Reconstruction Bands!', [b.shape for b in bands_rec])

    # Add losses.
    if training:
      # Audio losses.
      self._update_losses_dict(self.audio_losses, audio_padded, audio_rec)

      # Audio losses.
      self._update_losses_dict(self.band_losses, bands, bands_rec)

      # Apply GAN losses on each band.
      for loss, d_real, d_rec in zip(self.gan_losses, ds_real, ds_rec):
        self._update_losses_dict(loss, d_real, d_rec)

      # Committment losses.
      for q, z, z_q in zip(self.quantizers, zs, z_qs):
        self._update_losses_dict(q, z, z_q)

    outputs = {
        'bands': bands,
        'audio_padded': audio_padded,
        'zs': zs,
        'z_qs': z_qs,
        'cs': cs,
        'ds_real': ds_real,
        'ds_rec': ds_rec,
        'bands_rec': bands_rec,
        'audio_rec': audio_rec,
        'bands_rec_up': bands_rec_up,
    }
    return outputs

  def discriminate(self, bands, bands_rec):
    """Run discriminator on real and generated data."""
    ds_real = []
    ds_rec = []

    for disc, band, band_rec in zip(self.discriminators, bands, bands_rec):
      ds_real.append(disc(band[..., None]))  # Add back in channel dim.
      ds_rec.append(disc(band_rec[..., None]))  # Add back in channel dim.

    # Extract logits from output dictionary and remove channel dim.
    ds_real = [d['logits'][..., 0] for d in ds_real]
    ds_rec = [d['logits'][..., 0] for d in ds_rec]
    return ds_real, ds_rec

  @property
  def discriminator_variables(self):
    """Return all variables for discriminator networks."""
    d_vars = []
    for net in self.discriminators:
      d_vars += list(net.trainable_variables)
    return d_vars

  @property
  def generator_variables(self):
    """Return all variables for generator networks / decoders."""
    g_vars = []
    for net in self.decoders:
      g_vars += list(net.trainable_variables)
    return g_vars

  @property
  def other_variables(self):
    """Return both encoder and decoder variables."""
    gan_variables = self.discriminator_variables
    return self._set_difference(self.trainable_variables, gan_variables)

  def get_discriminator_losses(self, losses_dict):
    """Filter losses_dict and return losses for discriminator networks."""
    return {k: v for k, v in losses_dict.items() if k in self.d_loss_keys}

  def get_generator_losses(self, losses_dict):
    """Filter losses_dict and return losses for discriminator networks."""
    return {k: v for k, v in losses_dict.items() if k in self.g_loss_keys}
