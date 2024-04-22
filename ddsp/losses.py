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

"""Library of loss functions."""

import functools
from typing import Dict, Text

import crepe
from ddsp import core
from ddsp import dags
from ddsp import spectral_ops
from ddsp.core import hz_to_midi
from ddsp.core import safe_divide
from ddsp.core import tf_float32

import gin
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfkl = tf.keras.layers

# Define Types.
TensorDict = Dict[Text, tf.Tensor]


# ---------------------- Base Classes ------------------------------------------
class Loss(tfkl.Layer):
  """Base class. Duck typing: Losses just must implement get_losses_dict()."""

  def get_losses_dict(self, *args, **kwargs):
    """Returns a dictionary of losses for the model."""
    loss = self(*args, **kwargs)
    return {self.name: loss}


@gin.register
class LossGroup(dags.DAGLayer):
  """Compute a group of loss layers on an outputs dictionary."""

  def __init__(self, dag: dags.DAG, **kwarg_losses):
    """Constructor, completely configurable via gin.

    Args:
      dag: A list of loss names/instances, with keys to extract call() inputs
        from a dictionary, ex:

        ['module', ['input_key', ...]]

        'module': Loss module instance or string name of module. For example,
          'spectral_loss' would access the attribute `loss_group.spectral_loss`.
        'input_key': List of strings, nested keys in dictionary of dag outputs.

      **kwarg_losses: Losses to add to LossGroup. Each kwarg that is a Loss will
        be added as a property of the layer, so that it will be accessible as
        `loss_group.kwarg`. Also, other keras kwargs such as 'name' are split
        off before adding modules.
    """
    super().__init__(dag, **kwarg_losses)
    self.loss_names = self.module_names

  @property
  def losses(self):
    """Loss getter."""
    return [getattr(self, name) for name in self.loss_names]

  def call(self, outputs: TensorDict, **kwargs) -> TensorDict:
    """Get a dictionary of loss values from all the losses.

    Args:
      outputs: A dictionary of model output tensors to feed into the losses.
      **kwargs: Other kwargs for all the modules in the dag.

    Returns:
      A flat dictionary of losses {name: scalar}.
    """
    dag_outputs = super().call(outputs, **kwargs)
    loss_outputs = {}
    for k in self.loss_names:
      loss_outputs.update(dag_outputs[k])
    return loss_outputs

  def get_losses_dict(self, outputs, **kwargs):
    """Returns a dictionary of losses for the model, alias __call__."""
    return self(outputs, **kwargs)


# ---------------------- Losses ------------------------------------------------
def mean_difference(target, value, loss_type='L1', weights=None):
  """Common loss functions.

  Args:
    target: Target tensor.
    value: Value tensor.
    loss_type: One of 'L1', 'L2', or 'COSINE'.
    weights: A weighting mask for the per-element differences.

  Returns:
    The average loss.

  Raises:
    ValueError: If loss_type is not an allowed value.
  """
  difference = target - value
  weights = 1.0 if weights is None else weights
  loss_type = loss_type.upper()
  if loss_type == 'L1':
    return tf.reduce_mean(tf.abs(difference * weights))
  elif loss_type == 'L2':
    return tf.reduce_mean(difference**2 * weights)
  elif loss_type == 'COSINE':
    return tf.losses.cosine_distance(target, value, weights=weights, axis=-1)
  else:
    raise ValueError('Loss type ({}), must be '
                     '"L1", "L2", or "COSINE"'.format(loss_type))


@gin.register
class SpectralLoss(Loss):
  """Multi-scale spectrogram loss.

  This loss is the bread-and-butter of comparing two audio signals. It offers
  a range of options to compare spectrograms, many of which are redunant, but
  emphasize different aspects of the signal. By far, the most common comparisons
  are magnitudes (mag_weight) and log magnitudes (logmag_weight).
  """

  def __init__(self,
               fft_sizes=(2048, 1024, 512, 256, 128, 64),
               loss_type='L1',
               mag_weight=1.0,
               delta_time_weight=0.0,
               delta_freq_weight=0.0,
               cumsum_freq_weight=0.0,
               logmag_weight=0.0,
               loudness_weight=0.0,
               name='spectral_loss'):
    """Constructor, set loss weights of various components.

    Args:
      fft_sizes: Compare spectrograms at each of this list of fft sizes. Each
        spectrogram has a time-frequency resolution trade-off based on fft size,
        so comparing multiple scales allows multiple resolutions.
      loss_type: One of 'L1', 'L2', or 'COSINE'.
      mag_weight: Weight to compare linear magnitudes of spectrograms. Core
        audio similarity loss. More sensitive to peak magnitudes than log
        magnitudes.
      delta_time_weight: Weight to compare the first finite difference of
        spectrograms in time. Emphasizes changes of magnitude in time, such as
        at transients.
      delta_freq_weight: Weight to compare the first finite difference of
        spectrograms in frequency. Emphasizes changes of magnitude in frequency,
        such as at the boundaries of a stack of harmonics.
      cumsum_freq_weight: Weight to compare the cumulative sum of spectrograms
        across frequency for each slice in time. Similar to a 1-D Wasserstein
        loss, this hopefully provides a non-vanishing gradient to push two
        non-overlapping sinusoids towards eachother.
      logmag_weight: Weight to compare log magnitudes of spectrograms. Core
        audio similarity loss. More sensitive to quiet magnitudes than linear
        magnitudes.
      loudness_weight: Weight to compare the overall perceptual loudness of two
        signals. Very high-level loss signal that is a subset of mag and
        logmag losses.
      name: Name of the module.
    """
    super().__init__(name=name)
    self.fft_sizes = fft_sizes
    self.loss_type = loss_type
    self.mag_weight = mag_weight
    self.delta_time_weight = delta_time_weight
    self.delta_freq_weight = delta_freq_weight
    self.cumsum_freq_weight = cumsum_freq_weight
    self.logmag_weight = logmag_weight
    self.loudness_weight = loudness_weight

    self.spectrogram_ops = []
    for size in self.fft_sizes:
      spectrogram_op = functools.partial(spectral_ops.compute_mag, size=size)
      self.spectrogram_ops.append(spectrogram_op)

  def call(self, target_audio, audio, weights=None):
    loss = 0.0

    diff = core.diff
    cumsum = tf.math.cumsum

    # Compute loss for each fft size.
    for loss_op in self.spectrogram_ops:
      target_mag = loss_op(target_audio)
      value_mag = loss_op(audio)

      # Add magnitude loss.
      if self.mag_weight > 0:
        loss += self.mag_weight * mean_difference(
            target_mag, value_mag, self.loss_type, weights=weights)

      if self.delta_time_weight > 0:
        target = diff(target_mag, axis=1)
        value = diff(value_mag, axis=1)
        loss += self.delta_time_weight * mean_difference(
            target, value, self.loss_type, weights=weights)

      if self.delta_freq_weight > 0:
        target = diff(target_mag, axis=2)
        value = diff(value_mag, axis=2)
        loss += self.delta_freq_weight * mean_difference(
            target, value, self.loss_type, weights=weights)

      # TODO(kyriacos) normalize cumulative spectrogram
      if self.cumsum_freq_weight > 0:
        target = cumsum(target_mag, axis=2)
        value = cumsum(value_mag, axis=2)
        loss += self.cumsum_freq_weight * mean_difference(
            target, value, self.loss_type, weights=weights)

      # Add logmagnitude loss, reusing spectrogram.
      if self.logmag_weight > 0:
        target = spectral_ops.safe_log(target_mag)
        value = spectral_ops.safe_log(value_mag)
        loss += self.logmag_weight * mean_difference(
            target, value, self.loss_type, weights=weights)

    if self.loudness_weight > 0:
      target = spectral_ops.compute_loudness(target_audio, n_fft=2048,
                                             use_tf=True)
      value = spectral_ops.compute_loudness(audio, n_fft=2048, use_tf=True)
      loss += self.loudness_weight * mean_difference(
          target, value, self.loss_type, weights=weights)

    return loss


@gin.register
class HmmTranscriber(tfp.distributions.HiddenMarkovModel):
  """HMM initialized for decoding MIDI from Pitch and Amps."""

  def __init__(self,
               avg_length=200,
               midi_std=0.5,
               amps_on_center=1.5,
               amps_on_scale=0.5,
               amps_off_center=0.0,
               amps_off_scale=0.1,
               n_timesteps=1000,
               n_pitches=128,
               weight=1.0,
               **kwargs):
    """Discrete hidden states for each midi pitch, f0 observations (in midi).

    Args:
      avg_length: Prior over average note length between transitions.
      midi_std: Prior over f0 variance (in midi) allowed around discrete states.
      amps_on_center: Center amplitude of the "on" state.
      amps_on_scale: Variance amplitude of the "on" state.
      amps_off_center: Center amplitude of the "off" state.
      amps_off_scale: Variance amplitude of the "off" state.
      n_timesteps: Number of timesteps in the batch to unroll the HMM.
      n_pitches: Number of pitches (starting from 0) to use as HMM states.
      weight: Weighting of the nll loss term.
      **kwargs: Other kwargs for the distribution such as name.
    """
    # Initial distribution is uniform.
    initial_distribution = tfp.distributions.Categorical(
        probs=tf.ones([n_pitches]) / n_pitches)

    # Transition is heavily peaked around diagonal and uniform otherwise.
    hold = 1.0 - 1.0/avg_length
    other = (1.0 - hold) / (n_pitches - 1)
    transitions = ((hold - other) * tf.eye(n_pitches)
                   + other * tf.ones([n_pitches, n_pitches]))
    transitions /= tf.reduce_sum(transitions, axis=1, keepdims=True)
    transition_distribution = tfp.distributions.Categorical(
        probs=transitions)

    # Observations are normally distributed around the MIDI pitch (hmm state).
    p_loc = tf.range(1, n_pitches, dtype=tf.float32)
    p_scale = tf.ones([n_pitches - 1]) * midi_std
    pitch_loc = tf.concat([tf.ones([1]) * n_pitches / 2.0, p_loc], axis=0)
    pitch_scale = tf.concat([tf.ones([1]) * n_pitches, p_scale], axis=0)

    amps_loc = tf.concat([tf.ones([1]) * amps_off_center,
                          tf.ones(n_pitches - 1) * amps_on_center], axis=0)
    amps_scale = tf.concat([tf.ones([1]) * amps_off_scale,
                            tf.ones(n_pitches - 1) * amps_on_scale], axis=0)

    loc = tf.stack([pitch_loc, amps_loc], axis=-1)
    scale = tf.stack([pitch_scale, amps_scale], axis=-1)

    # observation_distribution = tfp.distributions.Normal(loc=loc, scale=scale)
    observation_distribution = tfp.distributions.MultivariateNormalDiag(
        loc=loc, scale_diag=scale)

    super().__init__(
        initial_distribution=initial_distribution,
        transition_distribution=transition_distribution,
        observation_distribution=observation_distribution,
        num_steps=n_timesteps,
        **kwargs
    )

    self.initial_distribution = initial_distribution
    self.transition_distribution = transition_distribution
    self.observation_distribution = observation_distribution
    self.avg_length = avg_length
    self.midi_std = midi_std
    self.n_timesteps = n_timesteps
    self.n_pitches = n_pitches
    self.weight = weight

  def __call__(self, pitch, amps):
    return self.nll(pitch, amps)

  @staticmethod
  def straight_through(x, x_quant):
    """Straight through estimation."""
    return x - tf.stop_gradient(x - x_quant)

  def nll(self, pitch, amps, per_example_loss=False):
    """Negative log-likelihood per a timestep."""
    pa = tf.concat([pitch, amps], axis=-1)
    avg_nll = -self.log_prob(pa) / pitch.shape[1]
    loss = avg_nll if per_example_loss else tf.reduce_mean(avg_nll)
    return self.weight * loss

  def predict_midi(self, pitch, amps, channel_dim=True, dtype=tf.float32):
    """Viterbi decode most likely hidden state as the quantized MIDI signal."""
    pa = tf.concat([pitch, amps], axis=-1)
    q_pitch = self.posterior_mode(pa)
    q_pitch = tf.cast(q_pitch, dtype)
    if channel_dim:
      q_pitch = q_pitch[:, :, tf.newaxis]
    return q_pitch




# ------------------------------------------------------------------------------
# Peceptual Losses
# ------------------------------------------------------------------------------
@gin.register
class EmbeddingLoss(Loss):
  """Embedding loss for a given pretrained model.

  Using these "perceptual" loss functions will help encourage better matching
  of higher-order structure than just a spectral loss. In image models, these
  perceptual losses help overcome the tendancy of straightforward L1 and L2
  losses to lead to blurry images. For ddsp, a "blurry" image is often a
  filtered noise synthesizer or reverb smearing power density in the right areas
  of a spectrogram but without the appropriate harmonic structure. This
  perceptual loss encouages output from harmonic and sinusoidal models because
  the pretrained model compares the two audio signals with features that are
  trained to detect the harmonic structure of natural sounds.
  """

  def __init__(self,
               weight=1.0,
               loss_type='L1',
               pretrained_model=None,
               name='embedding_loss'):
    super().__init__(name=name)
    self.weight = weight
    self.loss_type = loss_type
    self.pretrained_model = pretrained_model

  def call(self, target_audio, audio):
    loss = 0.0
    if self.weight > 0.0:
      audio, target_audio = tf_float32(audio), tf_float32(target_audio)
      target_emb = self.pretrained_model(target_audio)
      synth_emb = self.pretrained_model(audio)
      loss = self.weight * mean_difference(
          target_emb, synth_emb, self.loss_type)
    return loss


@gin.register
class PretrainedCREPEEmbeddingLoss(EmbeddingLoss):
  """Embedding loss of a pretrained CREPE model."""

  def __init__(self,
               weight=1.0,
               loss_type='L1',
               model_capacity='tiny',
               activation_layer='classifier',
               name='pretrained_crepe_embedding_loss'):
    # Scale each layer activation loss to comparable scales.
    scale = {
        'conv1-BN': 1.3,
        'conv1-maxpool': 1.0,
        'conv2-BN': 1.4,
        'conv2-maxpool': 1.1,
        'conv3-BN': 1.9,
        'conv3-maxpool': 1.6,
        'conv4-BN': 1.5,
        'conv4-maxpool': 1.4,
        'conv5-BN': 1.9,
        'conv5-maxpool': 1.7,
        'conv6-BN': 30,
        'conv6-maxpool': 25,
        'classifier': 130,
    }[activation_layer]
    super().__init__(
        weight=20.0 * scale * weight,
        loss_type=loss_type,
        name=name,
        pretrained_model=PretrainedCREPE(model_capacity=model_capacity,
                                         activation_layer=activation_layer))


class PretrainedCREPE(tfkl.Layer):
  """Pretrained CREPE model with frozen weights."""

  def __init__(self,
               model_capacity='tiny',
               activation_layer='conv5-maxpool',
               name='pretrained_crepe',
               trainable=False):
    super().__init__(name=name, trainable=trainable)
    self._model_capacity = model_capacity
    self._activation_layer = activation_layer
    spectral_ops.reset_crepe()
    self._model = crepe.core.build_and_load_model(self._model_capacity)
    self.frame_length = 1024

  def build(self, unused_x_shape):
    self.layer_names = [l.name for l in self._model.layers]

    if self._activation_layer not in self.layer_names:
      raise ValueError(
          'activation layer {} not found, valid names are {}'.format(
              self._activation_layer, self.layer_names))

    self._activation_model = tf.keras.Model(
        inputs=self._model.input,
        outputs=self._model.get_layer(self._activation_layer).output)

    # Variables are not to be trained.
    self._model.trainable = self.trainable
    self._activation_model.trainable = self.trainable

  def frame_audio(self, audio, hop_length=1024, center=True):
    """Slice audio into frames for crepe."""
    # Pad so that frames are centered around their timestamps.
    # (i.e. first frame is zero centered).
    pad = int(self.frame_length / 2)
    audio = tf.pad(audio, ((0, 0), (pad, pad))) if center else audio
    frames = tf.signal.frame(audio,
                             frame_length=self.frame_length,
                             frame_step=hop_length)

    # Normalize each frame -- this is expected by the model.
    mean, var = tf.nn.moments(frames, [-1], keepdims=True)
    frames -= mean
    frames /= (var**0.5 + 1e-5)
    return frames

  def call(self, audio):
    """Returns the embeddings.

    Args:
      audio: tensors of shape [batch, length]. Length must be divisible by 1024.

    Returns:
      activations of shape [batch, depth]
    """
    frames = self.frame_audio(audio)
    batch_size = int(frames.shape[0])
    n_frames = int(frames.shape[1])
    # Get model predictions.
    frames = tf.reshape(frames, [-1, self.frame_length])
    outputs = self._activation_model(frames)
    outputs = tf.reshape(outputs, [batch_size, n_frames, -1])
    return outputs


# ------------------------------------------------------------------------------
# Consistency Losses
# ------------------------------------------------------------------------------
def amp_loss(amp,
             amp_target,
             loss_type='L1',
             weights=None,
             log=False,
             amin=1e-5):
  """Loss comparing two amplitudes (scale logarithmically)."""
  if log:
    # Put in a log scale (psychophysically appropriate for audio).
    amp = core.log10(tf.maximum(amin, amp))
    amp_target = core.log10(tf.maximum(amin, amp_target))
  # Take the difference.
  return mean_difference(amp, amp_target, loss_type, weights)


def freq_loss(f_hz, f_hz_target, loss_type='L1', weights=None):
  """Loss comparing two frequencies."""
  # Convert to MIDI.
  f_midi = hz_to_midi(f_hz)
  f_midi_target = hz_to_midi(f_hz_target)
  # Take the difference.
  return mean_difference(f_midi, f_midi_target, loss_type, weights)


@gin.register
class FilteredNoiseConsistencyLoss(Loss):
  """Consistency loss for synthesizer controls.

  EXPERIMENTAL
  """

  def __init__(self, weight=1.0, **kwargs):
    super().__init__(**kwargs)
    self.weight = weight

  def call(self, noise_magnitudes, noise_magnitudes_target):
    """Add losses to the model."""
    self.built = True
    return self.weight * amp_loss(noise_magnitudes, noise_magnitudes_target)


@gin.register
class HarmonicConsistencyLoss(Loss):
  """Consistency loss for synthesizer controls.

  EXPERIMENTAL
  """

  def __init__(self,
               amp_weight=1.0,
               dist_weight=1.0,
               f0_weight=1.0,
               amp_threshold=1e-4,
               **kwargs):
    super().__init__(**kwargs)
    self.amp_weight = amp_weight
    self.dist_weight = dist_weight
    self.f0_weight = f0_weight
    self.amp_threshold = amp_threshold

  def call(self,
           harm_amp,
           harm_amp_target,
           harm_dist,
           harm_dist_target,
           f0_hz,
           f0_hz_target):
    """Add losses to the model."""
    self.built = True
    losses_dict = {}

    # Mask loss where target audio is below threshold amplitude.
    weights = tf.cast(harm_amp_target >= self.amp_threshold, tf.float32)

    # Harmonic amplitude.
    harm_amp_loss = amp_loss(harm_amp, harm_amp_target)
    losses_dict['harm_amp_loss'] = self.amp_weight * harm_amp_loss

    # Harmonic distribution.
    harm_dist_loss = amp_loss(harm_dist, harm_dist_target, weights=weights)
    losses_dict['harm_dist_loss'] = self.dist_weight * harm_dist_loss

    # Fundamental frequency.
    f0_hz_loss = freq_loss(f0_hz, f0_hz_target, weights=weights)
    losses_dict['f0_hz_loss'] = self.f0_weight * f0_hz_loss

    return losses_dict


# ------------------------------------------------------------------------------
# Sinusoidal Consistency Losses
# ------------------------------------------------------------------------------
@gin.register
class WassersteinConsistencyLoss(Loss):
  """Compare similarity of two traces of sinusoids using wasserstein distance.

  EXPERIMENTAL
  """

  def __init__(self,
               weight=1.0,
               midi=True,
               **kwargs):
    """Constructor.

    Args:
      weight: Loss weight.
      midi: Scale frequencies logarithmically (MIDI).
      **kwargs: Extra args for tfkl.Layer.
    """
    super().__init__(**kwargs)
    self.weight = weight
    self.midi = midi

  def call(self, amps_a, freqs_a, amps_b, freqs_b):
    """Returns the sinusoidal consistency loss scalar.

    Args:
      amps_a: Amplitudes of first sinusoids, greater than 0.
        Shape [batch, time, freq].
      freqs_a: Frequencies of first sinusoids in hertz.
        Shape [batch, time, feq].
      amps_b: Amplitudes of second sinusoids, greater than 0.
        Shape [batch, time, freq].
      freqs_b: Frequencies of second sinusoids in hertz.
        Shape [batch, time, feq].

    Returns:
      Scalar, weighted wasserstein distance.
    """
    loss = 0.0
    if self.weight > 0.0:
      if self.midi:
        freqs_a = hz_to_midi(freqs_a)
        freqs_b = hz_to_midi(freqs_b)
        loss = wasserstein_distance(freqs_a, freqs_b, amps_a, amps_b, p=1.0)
        loss = tf.reduce_mean(self.weight * loss)
    return loss


def wasserstein_distance(u_values, v_values, u_weights, v_weights, p=1.0):
  """Differentiable 1-D Wasserstein distance.

  Adapted from the scipy.stats implementation.
  Args:
    u_values: Samples from distribution `u`. Shape [batch_shape, n_samples].
    v_values: Samples from distribution `v`. Shape [batch_shape, n_samples].
    u_weights: Sample weights. Shape [batch_shape, n_samples].
    v_weights: Sample weights. Shape [batch_shape, n_samples].
    p: Degree of the distance norm. Wasserstein=1, Energy=2.

  Returns:
    The Wasserstein distance between samples. Shape [batch_shape].
  """
  u_sorter = tf.argsort(u_values, axis=-1)
  v_sorter = tf.argsort(v_values, axis=-1)

  all_values = tf.concat([u_values, v_values], axis=-1)
  all_values = tf.sort(all_values, axis=-1)

  # Compute the differences between pairs of successive values of u and v.
  deltas = core.diff(all_values, axis=-1)

  # Get the respective positions of the values of u and v among the values of
  # both distributions.
  batch_dims = len(u_values.shape) - 1
  gather = lambda x, i: tf.gather(x, i, axis=-1, batch_dims=batch_dims)
  u_cdf_indices = tf.searchsorted(
      gather(u_values, u_sorter), all_values[..., :-1], side='right')
  v_cdf_indices = tf.searchsorted(
      gather(v_values, v_sorter), all_values[..., :-1], side='right')

  # Calculate the CDFs of u and v using their weights, if specified.
  if u_weights is None:
    u_cdf = u_cdf_indices / float(u_values.shape[-1])
  else:
    u_sorted_cumweights = tf.concat(
        [tf.zeros_like(u_weights)[..., 0:1],
         tf.cumsum(gather(u_weights, u_sorter), axis=-1)],
        axis=-1)
    u_cdf = gather(u_sorted_cumweights, u_cdf_indices)
    safe_divide(u_cdf, u_sorted_cumweights[..., -1:])

  if v_weights is None:
    v_cdf = v_cdf_indices / float(v_values.shape[-1])
  else:
    v_sorted_cumweights = tf.concat(
        [tf.zeros_like(v_weights)[..., 0:1],
         tf.cumsum(gather(v_weights, v_sorter), axis=-1)],
        axis=-1)
    v_cdf = gather(v_sorted_cumweights, v_cdf_indices)
    safe_divide(v_cdf, v_sorted_cumweights[..., -1:])

  # Compute the value of the integral based on the CDFs.
  return tf.reduce_sum(deltas * tf.abs(u_cdf - v_cdf)**p, axis=-1)**(1.0 / p)


@gin.register
class KDEConsistencyLoss(Loss):
  """Compare similarity of two traces of sinusoids using kernels.

  EXPERIMENTAL

  Adapted from differentiable two-way mismatch loss. Very similar to the
  Jefferys divergence. Use gaussian kernel density estimate in both directions
  to compare likelihood of each set of sinusoids relative to the other.

  Also enforces mean amplitudes to be the same, as JD by itself is insensitve to
  absolute scale of the amplitudes.
  """

  def __init__(self,
               weight_a=1.0,
               weight_b=1.0,
               weight_mean_amp=1.0,
               scale_a=0.1,
               scale_b=0.1,
               **kwargs):
    """Constructor.

    Args:
      weight_a: Weight for -log p(a|b) term.
      weight_b: Weight for -log p(b|a) term.
      weight_mean_amp: Weight to match the mean amplitudes between a and b.
      scale_a: Scale of the gaussians around each sinusoid in MIDI.
      scale_b: Scale of the gaussians around each sinusoid in MIDI.
      **kwargs: Extra args for tfkl.Layer.
    """
    super().__init__(**kwargs)
    # Loss weights.
    self.weight_a = weight_a
    self.weight_b = weight_b
    self.weight_mean_amp = weight_mean_amp
    # Gaussian widths in MIDI.
    self.scale_a = scale_a
    self.scale_b = scale_b

  def call(self, amps_a, freqs_a, amps_b, freqs_b):
    """Returns the sinusoidal consistency loss scalar.

    Args:
      amps_a: Amplitudes of first sinusoids, greater than 0.
        Shape [batch, time, freq].
      freqs_a: Frequencies of first sinusoids in hertz.
        Shape [batch, time, feq].
      amps_b: Amplitudes of second sinusoids, greater than 0.
        Shape [batch, time, freq].
      freqs_b: Frequencies of second sinusoids in hertz.
        Shape [batch, time, feq].

    Returns:
      Scalar, weighted -log p(a|b) - log p(b|a).
    """
    loss = 0.0
    if self.weight_a > 0.0:
      loss_a = self.nll(amps_a, freqs_a, amps_b, freqs_b, self.scale_b)
      loss += tf.reduce_mean(self.weight_a * loss_a)
    if self.weight_b > 0.0:
      loss_b = self.nll(amps_b, freqs_b, amps_a, freqs_a, self.scale_a)
      loss += tf.reduce_mean(self.weight_b * loss_b)
    if self.weight_mean_amp > 0.0:
      mean_amp_a = tf.reduce_mean(amps_a, axis=-1)
      mean_amp_b = tf.reduce_mean(amps_b, axis=-1)
      loss_mean_amp = tf.reduce_mean(tf.abs(mean_amp_a - mean_amp_b))
      loss += self.weight_mean_amp * loss_mean_amp
    return loss

  def nll(self, amps, freqs, amps_target, freqs_target, scale_target):
    """Returns negative log-likelihood of source sins given target sins.

    Args:
      amps: Amplitudes of source sinusoids, greater than 0.
        Shape [batch, time, freq].
      freqs: Frequencies of source sinusoids in hertz.
        Shape [batch, time, feq].
      amps_target: Amplitudes of target sinusoids, greater than 0.
        Shape [batch, time, freq].
      freqs_target: Frequencies of target sinusoids in hertz.
        Shape [batch, time, feq].
      scale_target: Scale of gaussian kernel in MIDI.

    Returns:
      - log(p(source|target)). Shape [batch, time].
    """
    p_source_given_target = self.kernel_density_estimate(
        amps_target, freqs_target, scale_target)

    # KDE is on a logarithmic scale (MIDI).
    freqs_midi = hz_to_midi(freqs)

    # Need to rearrage shape as tfp expects, [sample_sh, batch_sh, event_sh].
    freqs_transpose = tf.transpose(freqs_midi, [2, 0, 1])  # [freq, batch, time]
    nll_transpose = - p_source_given_target.log_prob(freqs_transpose)
    nll = tf.transpose(nll_transpose, [1, 2, 0])  # [batch, time, freq]

    # Weighted sum over sinusoids -> [batch, time]
    amps_norm = safe_divide(amps, tf.reduce_sum(amps, axis=-1, keepdims=True))
    return tf.reduce_mean(nll * amps_norm, axis=-1)

  def kernel_density_estimate(self, amps, freqs, scale):
    """Gets distribution of harmonics from candidate f0s given sinusoids.

    Performs a gaussian kernel density estimate on the sinusoid points, with the
    height of each gaussian component given by the sinusoidal amplitude.
    Args:
      amps: Amplitudes of sinusoids, must be greater than 0.
      freqs: Frequencies of sinusoids in hertz.
      scale: Scale of gaussian kernel in MIDI.

    Returns:
      MixtureSameFamily, Gaussian distribution.
    """
    # Gaussian KDE around each partial, height=amplitude, center=frequency.
    freqs_midi = hz_to_midi(freqs)

    # NLL can be a nan if sinusoid amps are all zero, add a small offset.
    amps = tf.where(amps == 0.0, 1e-7 * tf.ones_like(amps), amps)
    amps_norm = safe_divide(amps, tf.reduce_sum(amps, axis=-1, keepdims=True))

    # P(candidate_harmonics | freqs)
    return tfd.MixtureSameFamily(tfd.Categorical(probs=amps_norm),
                                 tfd.Normal(loc=freqs_midi, scale=scale))


# ------------------------------------------------------------------------------
# Differentiable Two-way Mismatch Loss
# ------------------------------------------------------------------------------
@gin.register
class TWMLoss(Loss):
  """Two-way Mismatch, encourages sinusoids to be harmonics best f0 candidate.

  EXPERIMENTAL

  This loss function is a differentiable / smooth probabalistic adaptation of
  the heuristic Two-way Mismatch (TWM) algorithm used to extract a fundamental
  frequency from a group of sinusoids.

  Sketch of the algorithm:
    * Given f0 candidates, produce a harmonic series off each candidate.
    * Place a gaussian at each harmonic peak and evaluate the likelihood of the
        sinusoids given that harmonic distribution p(sinusoids|harmonics).
    * Place a gaussian at each sinusoid peak, and evaluate the likelihood of the
        harmonics given that sinusoidal distribution p(harmonics|sinusoids).
    * The total loss is the combined negative log-likelihood of each term,
        Loss = - log p(sinusoids|harmonics) - log p(harmonics|sinusoids), with a
        softmin over f0 candidates to only consider the best candidates.

  The two terms work against eachother, with p(sinusoids|harmonics) rewarding
  lower f0 candidates that have more densely packed coverage, and
  p(harmonics|sinusoids) rewarding higher f0 candidates that have less chance
  of falling outside the sinusoidal coverage. The global minima for most cases
  is an intermediate f0 where the harmonics and sinusoids align.

  This implementation can also be used (non-differentiably) for selecting the
  best f0 candidate using an argmin over the loss.

  Original paper:
  Maher, Beaucamp, Fundamental Frequency Estimation of Musical Signals using a
  two-way Mismatch Procedure, 1993.
  http://www.montana.edu/rmaher/publications/maher_jasa_0494_2254-2263.pdf

  Good tutorial video:
  https://www.coursera.org/lecture/audio-signal-processing/f0-detection-c7o1c
  """

  def __init__(self,
               sinusoids_weight=1.0,
               harmonics_weight=1.0,
               sinusoids_scale=0.5,
               harmonics_scale=0.2,
               n_harmonic_points=10,
               n_harmonic_gaussians=30,
               softmin_temperature=1.0,
               sample_rate=16000,
               **kwargs):
    """Constructor.

    Args:
      sinusoids_weight: Weight for -log p(sinusoids|harmonics) term.
      harmonics_weight: Weight for -log p(harmonics|sinusoids) term.
      sinusoids_scale: Scale of the gaussians around each sinusoid.
      harmonics_scale: Scale of the gaussians around each harmonic.
      n_harmonic_points: Number of points to consider for each candidate f0 in
        the p(harmonics|sinusoids) calculation.
      n_harmonic_gaussians: Number of gaussians for each candidate f0 in the
        p(sinusoids|harmonics) calculation.
      softmin_temperature: Greater than 0, lower temperatures focus more on just
         the best (loweset cost) f0 candidate for the total loss.
      sample_rate: Audio sample rate in hertz. Used for ignoring harmonics above
        nyquist.
      **kwargs: Extra args for tfkl.Layer.
    """
    super().__init__(**kwargs)
    self.softmin_temperature = softmin_temperature
    self.sample_rate = sample_rate

    # Loss weights.
    self.sinusoids_weight = sinusoids_weight
    self.harmonics_weight = harmonics_weight

    # p(sinusoids|harmonics).
    self.sinusoids_scale = sinusoids_scale
    self.n_harmonic_points = n_harmonic_points

    # p(harmonics|sinusoids).
    self.harmonics_scale = harmonics_scale
    self.n_harmonic_gaussians = n_harmonic_gaussians

  def call(self, f0_candidates, freqs, amps):
    """Returns the TWM loss scalar.

    Args:
      f0_candidates: Frequencies of candidates in hertz. [batch, time, freq].
      freqs: Frequencies of sinusoids in hertz. [batch, time, feq].
      amps: Amplitudes of sinusoids, must be greater than 0. [batch, time, feq].

    Returns:
      Scalar, weighted -log p(sinusoids|harmonics) - log p(harmonics|sinusoids),
        with a softmin over f0 candidates to just consider the best candidates.
    """
    sinusoids_loss, harmonics_loss = self.get_loss_tensors(
        f0_candidates, freqs, amps)

    # Loss is the minimum loss (loss for the best f0 candidate).
    # Use softmin to make it differentiable.
    combined_loss = (self.sinusoids_weight * sinusoids_loss +
                     self.harmonics_weight * harmonics_loss)
    softmin_loss = combined_loss * tf.nn.softmax(
        -combined_loss / self.softmin_temperature, axis=-1)
    return tf.reduce_mean(softmin_loss)

  def predict_f0(self, f0_candidates, freqs, amps):
    """Get the most likely f0 from the series of candidates.

    Args:
      f0_candidates: Frequencies of candidates in hertz. [batch, time, freq].
      freqs: Frequencies of sinusoids in hertz. [batch, time, feq].
      amps: Amplitudes of sinusoids, greater than 0. [batch, time, freq].

    Returns:
      f0_hz: Numpy array, most likely f0 among the candidates at each timestep.
        Shape [batch, time, 1].
    """
    sinusoids_loss, harmonics_loss = self.get_loss_tensors(
        f0_candidates, freqs, amps)
    loss = (self.sinusoids_weight * sinusoids_loss +
            self.harmonics_weight * harmonics_loss)
    # Argmin is not differentiable, switching to numpy, ignore nans for min.
    f0_idx = np.nanargmin(np.array(loss), axis=-1)[..., np.newaxis]
    f0_hz = np.take_along_axis(np.array(f0_candidates), f0_idx, axis=-1)
    return f0_hz

  def get_loss_tensors(self, f0_candidates, freqs, amps):
    """Get traces of loss to estimate fundamental frequency.

    Args:
      f0_candidates: Frequencies of candidates in hertz. [batch, time, freq].
      freqs: Frequencies of sinusoids in hertz. [batch, time, feq].
      amps: Amplitudes of sinusoids, greater than 0. [batch, time, freq].

    Returns:
      sinusoids_loss: -log p(sinusoids|harmonics), [batch, time, f0_candidate].
      harmonics_loss: - log p(harmonics|sinusoids), [batch, time, f0_candidate].
    """
    # ==========================================================================
    # P(sinusoids | candidate_harmonics).
    # ==========================================================================
    p_sinusoids_given_harmonics = self.get_p_sinusoids_given_harmonics()

    # Treat each partial as a candidate.
    # Get the ratio of each partial to each candidate.
    # -> [batch, time, candidate, partial]
    freq_ratios = safe_divide(freqs[:, :, tf.newaxis, :],
                              f0_candidates[:, :, :, tf.newaxis])
    nll_sinusoids = - p_sinusoids_given_harmonics.log_prob(freq_ratios)

    a = tf.convert_to_tensor(amps[:, :, tf.newaxis, :])

    # # Don't count sinusoids that are less than 1 std > mean.
    # a_mean, a_var = tf.nn.moments(a, axes=-1, keepdims=True)
    # a = tf.where(a > a_mean + 0.5 * a_var**0.5, a, tf.zeros_like(a))

    # Weighted sum by sinusoid amplitude.
    # -> [batch, time, candidate]
    sinusoids_loss = safe_divide(tf.reduce_sum(nll_sinusoids * a, axis=-1),
                                 tf.reduce_sum(a, axis=-1))

    # ==========================================================================
    # P(candidate_harmonics | sinusoids)
    # ==========================================================================
    p_harm_given_sin = self.get_p_harmonics_given_sinusoids(freqs, amps)
    harmonics = self.get_candidate_harmonics(f0_candidates, as_midi=True)

    # Need to rearrage shape as tfp expects, [sample_sh, batch_sh, event_sh].
    # -> [candidate, harmonic, batch, time]
    harmonics_transpose = tf.transpose(harmonics, [2, 3, 0, 1])
    nll_harmonics_transpose = - p_harm_given_sin.log_prob(harmonics_transpose)
    # -> [batch, time, candidate, harm]
    nll_harmonics = tf.transpose(nll_harmonics_transpose, [2, 3, 0, 1])

    # Prior decreasing importance of upper harmonics.
    amps_prior = tf.linspace(
        1.0, 1.0 / self.n_harmonic_points, self.n_harmonic_points)
    harmonics_loss = (nll_harmonics *
                      amps_prior[tf.newaxis, tf.newaxis, tf.newaxis, :])

    # Don't count loss for harmonics above nyquist.
    # Reweight by the number of harmonics below nyquist,
    # (so it doesn't just pick the highest frequency possible).
    nyquist_midi = hz_to_midi(self.sample_rate / 2.0)
    nyquist_mask = tf.where(harmonics < nyquist_midi,
                            tf.ones_like(harmonics_loss),
                            tf.zeros_like(harmonics_loss))
    harmonics_loss *= safe_divide(
        nyquist_mask, tf.reduce_mean(nyquist_mask, axis=-1, keepdims=True))

    # Sum over harmonics.
    harmonics_loss = tf.reduce_mean(harmonics_loss, axis=-1)

    return sinusoids_loss, harmonics_loss

  def get_p_sinusoids_given_harmonics(self):
    """Gets distribution of sinusoids given harmonics from candidate f0s.

    Returns:
      MixtureSameFamily, Gaussian distribution.
    """
    # Normalized frequency (harmonic number), create equally spaced gaussians.
    harmonics_probs = (tf.ones(self.n_harmonic_gaussians) /
                       self.n_harmonic_gaussians)
    harmonics_loc = tf.range(1, self.n_harmonic_gaussians + 1, dtype=tf.float32)

    # P(sinusoids | candidate_harmonics).
    return tfd.MixtureSameFamily(
        tfd.Categorical(harmonics_probs),
        tfd.Normal(loc=harmonics_loc, scale=self.harmonics_scale))

  def get_p_harmonics_given_sinusoids(self, freqs, amps):
    """Gets distribution of harmonics from candidate f0s given sinusoids.

    Performs a gaussian kernel density estimate on the sinusoid points, with the
    height of each gaussian component given by the sinusoidal amplitude.
    Args:
      freqs: Frequencies of sinusoids in hertz.
      amps: Amplitudes of sinusoids, must be greater than 0.

    Returns:
      MixtureSameFamily, Gaussian distribution.
    """
    # Gaussian KDE around each partial, height=amplitude, center=frequency.
    sinusoids_midi = hz_to_midi(freqs)

    # NLL can be a nan if sinusoid amps are all zero, add a small offset.
    amps = tf.where(amps == 0.0, 1e-7 * tf.ones_like(amps), amps)
    amps_norm = safe_divide(amps, tf.reduce_sum(amps, axis=-1, keepdims=True))

    # P(candidate_harmonics | sinusoids)
    return tfd.MixtureSameFamily(
        tfd.Categorical(probs=amps_norm),
        tfd.Normal(loc=sinusoids_midi, scale=self.sinusoids_scale))

  def get_candidate_harmonics(self, f0_candidates, as_midi=True):
    """Build a harmonic series off of each candidate partial."""
    n = tf.range(1, self.n_harmonic_points + 1, dtype=tf.float32)
    # -> [batch, time, candidate, harmonic]
    harmonics = (f0_candidates[:, :, :, tf.newaxis] *
                 n[tf.newaxis, tf.newaxis, tf.newaxis, :])
    if as_midi:
      harmonics = hz_to_midi(harmonics)
    return harmonics


@gin.register
class ParamLoss(Loss):
  """Loss on the mean difference between any two tensors."""

  def __init__(self, weight=1.0, loss_type='L1', **kwargs):
    super().__init__(**kwargs)
    self.weight = weight
    self.loss_type = loss_type

  def call(self, pred, target, weights=None):
    # Take the difference.
    loss = mean_difference(pred, target, self.loss_type, weights)
    return self.weight * loss


