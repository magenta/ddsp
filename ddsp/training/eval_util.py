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
"""Library of evaluation functions."""

import io
import os
import time

from absl import logging
import ddsp
from ddsp.core import tf_float32
import gin
import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v2 as tf

# Global values for evaluation.
MIN_F0_CONFIDENCE = 0.85
OUTLIER_MIDI_THRESH = 12


def squeeze(input_vector):
  """Ensure vector only has one axis of dimensionality."""
  if input_vector.ndim > 1:
    return np.squeeze(input_vector)
  else:
    return input_vector


# ---------------------- Metrics -----------------------------------------------
def l1_distance(prediction, ground_truth):
  """Average L1 distance difference between two 1-D vectors."""
  prediction, ground_truth = np.squeeze(prediction), np.squeeze(ground_truth)
  min_length = min(prediction.size, ground_truth.size)
  diff = prediction[:min_length] - ground_truth[:min_length]
  return np.abs(diff)


def is_outlier(ground_truth_f0_conf):
  """Determine if ground truth f0 for audio sample is an outlier."""
  ground_truth_f0_conf = squeeze(ground_truth_f0_conf)
  return np.max(ground_truth_f0_conf) < MIN_F0_CONFIDENCE


def compute_audio_features(audio,
                           n_fft=2048,
                           sample_rate=16000,
                           frame_rate=250):
  """Compute features from audio."""
  audio_feats = {'audio': audio}
  audio = squeeze(audio)

  audio_feats['loudness_db'] = ddsp.spectral_ops.compute_loudness(
      audio, sample_rate, frame_rate, n_fft)

  audio_feats['f0_hz'], audio_feats['f0_confidence'] = (
      ddsp.spectral_ops.compute_f0(audio, sample_rate, frame_rate))

  return audio_feats


def f0_dist_conf_thresh(f0_hz,
                        f0_hz_gen,
                        f0_confidence,
                        f0_confidence_thresh=MIN_F0_CONFIDENCE):
  """Compute L1 between gen audio and ground truth audio.

  Calculating F0 distance is more complicated than calculating loudness
  distance because of inherent inaccuracies in pitch tracking.

  We take the following steps:
  - Define a `keep_mask` that only select f0 values above when f0_confidence in
  the GENERATED AUDIO (not ground truth) exceeds a minimum threshold.
  Experimentation by jessengel@ and hanoih@ found this to be optimal way to
  filter out bad f0 pitch tracking.
  - Compute `delta_f0` between generated audio and ground truth audio.
  - Only select values in `delta_f0` based on this `keep_mask`
  - Compute mean on this selection
  - At the start of training, audio samples will sound bad and thus have no
  pitch content. If the `f0_confidence` is all below the threshold, we keep a
  count of it. A better performing model will have a smaller count of
  "untrackable pitch" samples.

  Args:
    f0_hz: Ground truth audio f0 in hertz [MB,:].
    f0_hz_gen: Generated audio f0 in hertz [MB,:].
    f0_confidence: Ground truth audio f0 confidence [MB,:]
    f0_confidence_thresh: Confidence threshold above which f0 metrics will be
      computed

  Returns:
    delta_f0_mean: Float or None if entire generated sample had
      f0_confidence below threshold. In units of MIDI (logarithmic frequency).
  """

  if np.max(f0_confidence) < f0_confidence_thresh:
    # Generated audio is not good enough for reliable pitch tracking.
    return None
  else:
    keep_mask = f0_confidence >= f0_confidence_thresh

    # Report mean error in midi space for easier interpretation.
    f0_midi = librosa.core.hz_to_midi(f0_hz)
    f0_midi_gen = librosa.core.hz_to_midi(f0_hz_gen)
    # Set -infs introduced by hz_to_midi to 0.
    f0_midi[f0_midi == -np.inf] = 0
    f0_midi_gen[f0_midi_gen == -np.inf] = 0

    delta_f0_midi = l1_distance(f0_midi, f0_midi_gen)
    delta_f0_midi_filt = delta_f0_midi[keep_mask]
    return np.mean(delta_f0_midi_filt)


class F0LoudnessMetrics(object):
  """Helper object for computing f0 and loudness metrics."""

  def __init__(self):
    self.metrics = {
        'loudness_db': tf.keras.metrics.Mean('loudness_db'),
        'f0_encoder': tf.keras.metrics.Mean('f0_encoder'),
        'f0_crepe': tf.keras.metrics.Mean('f0_crepe'),
        'f0_crepe_outlier_ratio':
            tf.keras.metrics.Accuracy('f0_crepe_outlier_ratio'),
    }

  def update_state(self, batch, audio_gen, f0_hz_predict):
    """Update metrics based on a batch of audio.

    Args:
      batch: Dictionary of input features.
      audio_gen: Batch of generated audio.
      f0_hz_predict: Batch of encoded f0, same as input f0 if no f0 encoder.
    """
    batch_size = int(audio_gen.shape[0])
    # Compute metrics per sample. No batch operations possible.
    for i in range(batch_size):
      # Extract features from generated audio example.
      keys = ['loudness_db', 'f0_hz', 'f0_confidence']
      feats = {k: v[i] for k, v in batch.items() if k in keys}
      feats_gen = compute_audio_features(audio_gen[i])

      # Loudness metric.
      ld_dist = np.mean(l1_distance(feats['loudness_db'],
                                    feats_gen['loudness_db']))
      self.metrics['loudness_db'].update_state(ld_dist)

      # F0 metric.
      if is_outlier(feats['f0_confidence']):
        # Ground truth f0 was unreliable to begin with. Discard.
        f0_crepe_dist = None
      else:
        # Gound truth f0 was reliable, compute f0 distance with generated audio
        f0_crepe_dist = f0_dist_conf_thresh(feats['f0_hz'],
                                            feats_gen['f0_hz'],
                                            feats['f0_confidence'])

        # Compute distance original f0_hz labels and f0 encoder values.
        # Resample if f0 encoder has different number of time steps.
        f0_encoder = f0_hz_predict[i]
        f0_original = feats['f0_hz']
        if f0_encoder.shape[0] != f0_original.shape[0]:
          f0_encoder = ddsp.core.resample(f0_encoder, f0_original.shape[0])
        f0_encoder_dist = f0_dist_conf_thresh(f0_original,
                                              f0_encoder,
                                              feats['f0_confidence'])
        self.metrics['f0_encoder'].update_state(f0_encoder_dist)

        if f0_crepe_dist is None or f0_crepe_dist > OUTLIER_MIDI_THRESH:
          # Generated audio had untrackable pitch content or is an outlier.
          self.metrics['f0_crepe_outlier_ratio'].update_state(True, True)
          logging.info('Sample %d has untrackable pitch content', i)
        else:
          # Generated audio had trackable pitch content and is within tolerance
          self.metrics['f0_crepe'].update_state(f0_crepe_dist)
          self.metrics['f0_crepe_outlier_ratio'].update_state(True, False)
          logging.info(
              'sample {} | ld_dist(db): {:.3f} | f0_crepe_dist(midi): {:.3f} | '
              'f0_encoder_dist(midi): {:.3f}'.format(
                  i, ld_dist, f0_crepe_dist, f0_encoder_dist))

  def flush(self, step):
    """Add summaries for each metric and reset the state."""
    # Start by logging the metrics result.
    logging.info('COMPUTING METRICS COMPLETE. FLUSHING ALL METRICS')
    metrics_str = ' | '.join(
        '{}: {:0.3f}'.format(k, v.result()) for k, v in self.metrics.items())
    logging.info(metrics_str)

    for name, metric in self.metrics.items():
      tf.summary.scalar('metrics/{}'.format(name), metric.result(), step)
      metric.reset_states()

    ddsp.spectral_ops.reset_crepe()  # Reset CREPE global state


# ---------------------- Custom summaries --------------------------------------
def fig_summary(tag, fig, step):
  """Writes an image summary from a string buffer of an mpl figure.

  This writer writes a scalar summary in V1 format using V2 API.

  Args:
    tag: An arbitrary tag name for this summary.
    fig: A matplotlib figure.
    step: The `int64` monotonic step variable, which defaults
      to `tf.compat.v1.train.get_global_step`.
  """
  buffer = io.BytesIO()
  fig.savefig(buffer, format='png')
  image_summary = tf.compat.v1.Summary.Image(
      encoded_image_string=buffer.getvalue())
  plt.close(fig)

  pb = tf.compat.v1.Summary()
  pb.value.add(tag=tag, image=image_summary)
  serialized = tf.convert_to_tensor(pb.SerializeToString())
  tf.summary.experimental.write_raw_pb(serialized, step=step, name=tag)


def waveform_summary(audio, audio_gen, step, name=''):
  """Creates a waveform plot summary for a batch of audio."""

  def plot_waveform(i, length=None, prefix='waveform', name=''):
    """Plots a waveforms."""
    waveform = squeeze(audio[i])
    waveform = waveform[:length] if length is not None else waveform
    waveform_gen = squeeze(audio_gen[i])
    waveform_gen = waveform_gen[:length] if length is not None else waveform_gen
    # Manually specify exact size of fig for tensorboard
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(2.5, 2.5))
    ax0.plot(waveform)
    ax1.plot(waveform_gen)

    # Format and save plot to image
    name = name + '_' if name else ''
    tag = 'waveform/{}{}_{}'.format(name, prefix, i + 1)
    fig_summary(tag, fig, step)

  # Make plots at multiple lengths.
  batch_size = int(audio.shape[0])
  for i in range(batch_size):
    plot_waveform(i, length=None, prefix='full', name=name)
    plot_waveform(i, length=2000, prefix='125ms', name=name)


def get_spectrogram(audio, rotate=False, size=1024):
  """Compute logmag spectrogram."""
  mag = ddsp.spectral_ops.compute_logmag(tf_float32(audio), size=size)
  if rotate:
    mag = np.rot90(mag)
  return mag


def spectrogram_summary(audio, audio_gen, step, name=''):
  """Writes a summary of spectrograms for a batch of images."""
  specgram = lambda a: ddsp.spectral_ops.compute_logmag(tf_float32(a), size=768)

  # Batch spectrogram operations
  spectrograms = specgram(audio)
  spectrograms_gen = specgram(audio_gen)

  batch_size = int(audio.shape[0])
  for i in range(batch_size):
    # Manually specify exact size of fig for tensorboard
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    ax = axs[0]
    spec = np.rot90(spectrograms[i])
    ax.matshow(spec, vmin=-5, vmax=1, aspect='auto', cmap=plt.cm.magma)
    ax.set_title('original')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[1]
    spec = np.rot90(spectrograms_gen[i])
    ax.matshow(spec, vmin=-5, vmax=1, aspect='auto', cmap=plt.cm.magma)
    ax.set_title('synthesized')
    ax.set_xticks([])
    ax.set_yticks([])

    # Format and save plot to image
    name = name + '_' if name else ''
    tag = 'spectrogram/{}{}'.format(name, i + 1)
    fig_summary(tag, fig, step)


def audio_summary(audio, step, sample_rate=16000, name='audio'):
  """Update metrics dictionary given a batch of audio."""
  # Ensure there is a single channel dimension.
  batch_size = int(audio.shape[0])
  if len(audio.shape) == 2:
    audio = audio[:, :, tf.newaxis]
  tf.summary.audio(
      name, audio, sample_rate, step, max_outputs=batch_size, encoding='wav')


def f0_summary(f0_hz, f0_hz_predict, step, name=''):
  """Creates a plot comparison of ground truth f0_hz and predicted values."""
  batch_size = int(f0_hz.shape[0])

  for i in range(batch_size):
    f0_midi = ddsp.core.hz_to_midi(squeeze(f0_hz[i]))
    f0_midi_predict = ddsp.core.hz_to_midi(squeeze(f0_hz_predict[i]))

    # Resample if f0_encoder has different number of time steps
    if f0_midi_predict.shape[0] != f0_midi.shape[0]:
      f0_midi_predict = ddsp.core.resample(f0_midi_predict, f0_midi.shape[0])

    # Manually specify exact size of fig for tensorboard
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.0, 2.0))
    ax0.plot(f0_midi)
    ax0.plot(f0_midi_predict)
    ax0.set_title('original vs. predicted')

    ax1.plot(f0_midi_predict)
    ax1.set_title('predicted')

    # Format and save plot to image
    name = name + '_' if name else ''
    tag = 'f0_midi/{}{}'.format(name, i + 1)
    fig_summary(tag, fig, step)


# ---------------------- Evaluation --------------------------------------------
def evaluate_or_sample(data_provider,
                       model,
                       mode='eval',
                       model_dir='~/tmp/ddsp/training',
                       batch_size=32,
                       num_batches=50,
                       ckpt_delay_secs=0,
                       run_once=False):
  """Run evaluation loop.

  Args:
    data_provider: DataProvider instance.
    model: Model instance.
    mode: Whether to 'eval' with metrics or create 'sample' s.
    model_dir: Path to directory with checkpoints and summary events.
    batch_size: Size of each eval/sample batch.
    num_batches: How many batches to eval from dataset. -1 denotes all batches.
    ckpt_delay_secs: Time to wait when a new checkpoint was not detected.
    run_once: Only run evaluation or sampling once.
  """
  # Set up the summary writer and metrics.
  summary_dir = os.path.join(model_dir, 'summaries', 'eval')
  summary_writer = tf.summary.create_file_writer(summary_dir)

  # Sample continuously and load the newest checkpoint each time
  checkpoints_iterator = tf.train.checkpoints_iterator(model_dir,
                                                       ckpt_delay_secs)

  # Get the dataset.
  dataset = data_provider.get_batch(batch_size=batch_size,
                                    shuffle=False,
                                    repeats=-1)

  with summary_writer.as_default():
    for checkpoint_path in checkpoints_iterator:
      step = int(checkpoint_path.split('-')[-1])

      # Redefine thte dataset iterator each time to make deterministic.
      dataset_iter = iter(dataset)

      # Load model.
      model.restore(checkpoint_path)

      # Iterate through dataset and make predictions
      checkpoint_start_time = time.time()

      for batch_idx in range(1, num_batches + 1):
        try:
          start_time = time.time()
          logging.info('Predicting batch %d of size %d', batch_idx, batch_size)

          # Predict a batch of audio.
          batch = next(dataset_iter)
          audio = batch['audio']
          # TODO(jesseengel): Find a way to add losses with training=False.
          audio_gen, losses = model(batch, return_losses=True, training=True)
          outputs = model.get_controls(batch, training=True)

          # Create metrics on first batch.
          if mode == 'eval' and batch_idx == 1:
            f0_loudness_metrics = F0LoudnessMetrics()
            avg_losses = {
                name: tf.keras.metrics.Mean(name=name, dtype=tf.float32)
                for name in list(losses.keys())}


          # Resample f0_hz outputs to match batch if they don't already.
          has_f0 = ('f0_hz' in outputs and 'f0_hz' in batch)
          if has_f0:
            output_length = outputs['f0_hz'].shape[1]
            batch_length = batch['f0_hz'].shape[1]
            if output_length != batch_length:
              outputs['f0_hz'] = ddsp.core.resample(
                  outputs['f0_hz'], batch_length)

          logging.info('Prediction took %.1f seconds', time.time() - start_time)

          if mode == 'sample':
            start_time = time.time()
            logging.info('Writing summmaries for batch %d', batch_idx)

            # Add audio.
            audio_summary(audio_gen, step, name='audio_generated')
            audio_summary(audio, step, name='audio_original')

            # Add plots.
            waveform_summary(audio, audio_gen, step)
            spectrogram_summary(audio, audio_gen, step)
            if has_f0:
              f0_summary(batch['f0_hz'], outputs['f0_hz'], step)

            logging.info('Writing batch %i with size %i took %.1f seconds',
                         batch_idx, batch_size, time.time() - start_time)

          elif mode == 'eval':
            start_time = time.time()
            logging.info('Calculating metrics for batch %d', batch_idx)

            if has_f0:
              # F0 and loudness.
              f0_loudness_metrics.update_state(batch,
                                               audio_gen,
                                               outputs['f0_hz'])

            # Loss.
            for k, v in losses.items():
              avg_losses[k].update_state(v)

            logging.info('Metrics for batch %i with size %i took %.1f seconds',
                         batch_idx, batch_size, time.time() - start_time)

        except tf.errors.OutOfRangeError:
          logging.info('End of dataset.')
          break

      logging.info('All %d batches in checkpoint took %.1f seconds',
                   num_batches, time.time() - checkpoint_start_time)

      if mode == 'eval':
        if has_f0:
          f0_loudness_metrics.flush(step)
        for k, metric in avg_losses.items():
          tf.summary.scalar('losses/{}'.format(k), metric.result(), step=step)
          metric.reset_states()

      summary_writer.flush()

      if run_once:
        break


@gin.configurable
def evaluate(data_provider,
             model,
             model_dir='~/tmp/ddsp/training',
             batch_size=32,
             num_batches=50,
             ckpt_delay_secs=0,
             run_once=False):
  """Run evaluation loop.

  Args:
    data_provider: DataProvider instance.
    model: Model instance.
    model_dir: Path to directory with checkpoints and summary events.
    batch_size: Size of each eval/sample batch.
    num_batches: How many batches to eval from dataset. -1 denotes all batches.
    ckpt_delay_secs: Time to wait when a new checkpoint was not detected.
    run_once: Only run evaluation or sampling once.
  """
  evaluate_or_sample(
      data_provider=data_provider,
      model=model,
      mode='eval',
      model_dir=model_dir,
      batch_size=batch_size,
      num_batches=num_batches,
      ckpt_delay_secs=ckpt_delay_secs,
      run_once=run_once)


@gin.configurable
def sample(data_provider,
           model,
           model_dir='~/tmp/ddsp/training',
           batch_size=16,
           num_batches=1,
           ckpt_delay_secs=0,
           run_once=False):
  """Run sampling loop.

  Args:
    data_provider: DataProvider instance.
    model: Model instance.
    model_dir: Path to directory with checkpoints and summary events.
    batch_size: Size of each eval/sample batch.
    num_batches: How many batches to eval from dataset. -1 denotes all batches.
    ckpt_delay_secs: Time to wait when a new checkpoint was not detected.
    run_once: Only run evaluation or sampling once.
  """
  evaluate_or_sample(
      data_provider=data_provider,
      model=model,
      mode='sample',
      model_dir=model_dir,
      batch_size=batch_size,
      num_batches=num_batches,
      ckpt_delay_secs=ckpt_delay_secs,
      run_once=run_once)
