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
from ddsp import spectral_ops
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

  audio_feats['loudness_db'] = spectral_ops.compute_loudness(
      audio, sample_rate, frame_rate, n_fft)

  audio_feats['f0_hz'], audio_feats['f0_confidence'] = spectral_ops.compute_f0(
      audio, sample_rate, frame_rate)

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
        'f0_midi': tf.keras.metrics.Mean('f0_midi'),
        'f0_outlier_ratio': tf.keras.metrics.Accuracy('f0_outlier_ratio')
    }

  def update_state(self, batch, audio_gen):
    """Update metrics based on a batch of audio."""
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
        f0_dist = None
      else:
        # Gound truth f0 was reliable, compute f0 distance with generated audio
        f0_dist = f0_dist_conf_thresh(feats['f0_hz'],
                                      feats_gen['f0_hz'],
                                      feats['f0_confidence'])

        if f0_dist is None or f0_dist > OUTLIER_MIDI_THRESH:
          # Generated audio had untrackable pitch content or is an outlier.
          self.metrics['f0_outlier_ratio'].update_state(True, True)

          logging.info('Sample %d has untrackable pitch content', i)
        else:
          # Generated audio had trackable pitch content and is within tolerance
          self.metrics['f0_midi'].update_state(f0_dist)
          self.metrics['f0_outlier_ratio'].update_state(True, False)

          logging.info('sample {} | ld_dist(db): {:.3f} '
                       '| f0_dist(midi): {:.3f}'.format(i, ld_dist, f0_dist))

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

    spectral_ops.reset_crepe()  # Reset CREPE global state


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


def waveform_summary(name, audio, step):
  """Creates a waveform plot summary for a batch of audio."""

  def plot_waveform(sample_idx, length=None, prefix='waveform'):
    """Plots a waveforms."""
    waveform = squeeze(audio[sample_idx])
    waveform = waveform[:length] if length is not None else waveform
    # Manually specify exact size of fig for tensorboard
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
    ax.plot(waveform)

    # Format and save plot to image
    tag = 'waveform/{}_{}_{}'.format(name, prefix, sample_idx)
    fig_summary(tag, fig, step)

  batch_size = int(audio.shape[0])
  for sample_idx in range(batch_size):
    plot_waveform(sample_idx, length=None, prefix='full')
    plot_waveform(sample_idx, length=2000, prefix='125ms')


def get_spectrogram(audio, rotate=False, size=1024):
  """Compute logmag spectrogram."""
  mag = spectral_ops.compute_logmag(tf_float32(audio), size=size)
  if rotate:
    mag = np.rot90(mag)
  return mag


def spectrogram_summary(name, audio, audio_gen, step):
  """Writes a summary of spectrograms for a batch of images."""
  specgram = lambda a: spectral_ops.compute_logmag(tf_float32(a), size=768)

  # Batch spectrogram operations
  spectrograms = specgram(audio)
  spectrograms_gen = specgram(audio_gen)

  batch_size = int(audio.shape[0])
  for sample_idx in range(batch_size):
    # Manually specify exact size of fig for tensorboard
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    ax = axs[0]
    spec = np.rot90(spectrograms[sample_idx])
    ax.matshow(spec, vmin=-5, vmax=1, aspect='auto', cmap=plt.cm.magma)
    ax.set_title('original')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[1]
    spec = np.rot90(spectrograms_gen[sample_idx])
    ax.matshow(spec, vmin=-5, vmax=1, aspect='auto', cmap=plt.cm.magma)
    ax.set_title('synthesized')
    ax.set_xticks([])
    ax.set_yticks([])

    # Format and save plot to image
    tag = 'spectrogram/{}_{}'.format(name, sample_idx)
    fig_summary(tag, fig, step)


def audio_summary(name, audio, step, sample_rate=16000):
  """Update metrics dictionary given a batch of audio."""
  # Ensure there is a single channel dimension.
  batch_size = int(audio.shape[0])
  if len(audio.shape) == 2:
    audio = audio[:, :, tf.newaxis]
  tf.summary.audio(
      name, audio, sample_rate, step, max_outputs=batch_size, encoding='wav')


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

  with summary_writer.as_default():
    for checkpoint_path in checkpoints_iterator:
      step = int(checkpoint_path.split('-')[-1])

      # Load model.
      model.restore(checkpoint_path)

      # Create metrics.
      if mode == 'eval':
        f0_loudness_metrics = F0LoudnessMetrics()
        avg_losses = {name: tf.keras.metrics.Mean(name=name, dtype=tf.float32)
                      for name in model.loss_names}

      # Redefine thte dataset each time to make deterministic.
      dataset = data_provider.get_batch(
          batch_size=batch_size, shuffle=False, repeats=-1)
      dataset_iter = iter(dataset)

      # Iterate through dataset and make predictions
      checkpoint_start_time = time.time()
      for batch_idx in range(1, num_batches + 1):
        try:
          start_time = time.time()
          logging.info('Predicting batch %d of size %d', batch_idx, batch_size)

          batch = next(dataset_iter)
          audio = batch['audio']
          audio_gen = model(batch)

          logging.info('Prediction took %.1f seconds', time.time() - start_time)

          if mode == 'sample':
            start_time = time.time()
            logging.info('Writing summmaries for batch %d', batch_idx)

            audio_summary('audio', audio, step)
            audio_summary('audio_gen', audio_gen, step)
            waveform_summary('audio', audio, step)
            waveform_summary('audio_gen', audio_gen, step)
            spectrogram_summary('spectrogram', audio, audio_gen, step)

            logging.info('Writing batch %i with size %i took %.1f seconds',
                         batch_idx, batch_size, time.time() - start_time)

          elif mode == 'eval':
            start_time = time.time()
            logging.info('Calculating metrics for batch %d', batch_idx)

            # F0 and loudness.
            f0_loudness_metrics.update_state(batch, audio_gen)

            # Loss.
            losses = model.losses_dict
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
