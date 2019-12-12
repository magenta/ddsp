# Copyright 2019 The DDSP Authors.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import io
import itertools
import os
import time

from absl import logging
import crepe
import ddsp
import gin
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import tensorflow.compat.v1 as tf

# Global values for NSynth evaluation.
AUDIO_CONFIG = {'n_fft': 2048, 'hop_length': 64, 'sample_rate': 16000}
MIN_F0_CONFIDENCE = 0.85
OUTLIER_MIDI_THRESH = 12

MAX_MIDI_PITCH = 127

# Mean and std for each eps calculated from 5000 samples.
EPS_CENTER_STATS = {1e-1: {'mean': -1.4405, 'std': 0.9490},
                    1e-4: {'mean': -4.0599, 'std': 3.5105}}


# ---------------------- midi2wave WaveRNN L1 Metrics -------------------------
def _center_val(value, mean, std):
  """Center value using mean and std."""
  return (value - mean) / std


def _log_scaling(value, eps):
  """Log scaling using eps as noise floor.

  Args:
    value: input value
    eps: "noise floor". 1e-4 expands quiet portions of signals
  Returns:
    log scaled value
  """
  return np.log(value + eps)


def check_and_squeeze_to_vector(input_vector):
  """Ensure vector only has one axis of dimensionality."""
  if input_vector.ndim > 1:
    return np.squeeze(input_vector)
  else:
    return input_vector


def l1_distance(prediction, ground_truth):
  """L1 distance difference between two vectors."""
  if prediction.shape != ground_truth.shape:
    prediction, ground_truth = np.squeeze(prediction), np.squeeze(ground_truth)
  assert prediction.shape == ground_truth.shape
  return np.abs(prediction - ground_truth)


def mean_l1_loudness(prediction, ground_truth):
  """Mean L1 loudness across a single sample."""
  if prediction.shape != ground_truth.shape:
    prediction, ground_truth = np.squeeze(prediction), np.squeeze(ground_truth)
  assert prediction.shape == ground_truth.shape
  dist = l1_distance(prediction, ground_truth)
  return np.mean(dist)


def compute_loudness(input_matrix,
                     input_type,
                     n_fft,
                     hop_length,
                     sr,
                     eps=1e-4,
                     center_data=True):
  """Compute centered loudness.

  Perceptual loudness is more representative of amplitude envelope than direct
  calculation of RMS energy. Note that output is not normalized to 0 to 1 to
  reduce effects of outliers.

  Args:
    input_matrix: audio or stft input
    input_type: "audio" or "stft"
    n_fft: number of fft bins
    hop_length: hop length
    sr: sample rate
    eps: epsilon or "noise floor".
    center_data: whether to center data using mean and std

  Returns:
    feature vector of shape [length]
  """
  if input_type == 'stft':
    # use stft as is
    stft = input_matrix
  elif input_type == 'audio':
    # compute stft from audio
    stft = librosa.stft(input_matrix, n_fft=n_fft, hop_length=hop_length)
  db_loudness = librosa.perceptual_weighting(
      np.abs(stft)**2,
      librosa.fft_frequencies(sr, n_fft),
      ref=1.0,
      amin=1e-15,
      top_db=200.0)
  loudness = np.mean(
      librosa.db_to_amplitude(db_loudness),
      axis=0)  # mean across freq in linear scale
  loudness = _log_scaling(loudness, eps=eps)
  if center_data:
    loudness = _center_val(
        loudness,
        mean=EPS_CENTER_STATS[eps]['mean'],
        std=EPS_CENTER_STATS[eps]['std'])
  return loudness


def compute_f0(audio, hop_length, sr, normalize_midi=True, viterbi=True):
  """Use CREPE to predict f0 and f0_confidence."""
  _, f0_hz, f0_confidence, _ = crepe.predict(
      audio, sr, viterbi=viterbi, step_size=hop_length / sr * 1000, verbose=0)
  f0 = librosa.core.hz_to_midi(f0_hz)
  if normalize_midi:
    f0 /= MAX_MIDI_PITCH
  # Set -infs introduced by hz_to_midi to 0.
  f0[f0 == -np.inf] = 0
  # Set nans to 0 in confidence.
  f0_confidence = np.nan_to_num(f0_confidence)
  return f0, f0_confidence


def compute_audio_features(audio, audio_config=None):
  """Compute features from audio."""
  if audio_config is None:
    audio_config = AUDIO_CONFIG
  audio_feats = {'audio': audio}
  audio = check_and_squeeze_to_vector(audio)
  audio_feats['loudness'] = compute_loudness(audio, 'audio',
                                             audio_config['n_fft'],
                                             audio_config['hop_length'],
                                             audio_config['sample_rate'])

  audio_feats['f0'], audio_feats['f0_confidence'] = compute_f0(
      audio,
      audio_config['hop_length'],
      audio_config['sample_rate'],
      normalize_midi=True)
  return audio_feats


def is_outlier(ground_truth_f0_conf):
  """Determine if ground truth f0 for audio sample is an outlier."""
  ground_truth_f0_conf = check_and_squeeze_to_vector(
      ground_truth_f0_conf)
  return np.max(ground_truth_f0_conf) < MIN_F0_CONFIDENCE


def f0_dist_conf_thresh(gen_f0,
                        gen_f0_confidence,
                        ground_truth_f0,
                        ground_truth_f0_confidence,
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
    gen_f0: generated audio f0 [MB,:]
    gen_f0_confidence: generated audio f0 confidence [MB,:]
    ground_truth_f0: ground truth audio f0 confidence [MB,:]
    ground_truth_f0_confidence: ground truth audio f0 confidence [MB,:]
    f0_confidence_thresh: confidence threshold above which f0 metrics will be
      computed

  Returns:
    delta_f0_mean: float or None if entire generated sample had
    f0_confidence below threshold.
  """

  if np.max(gen_f0_confidence) < f0_confidence_thresh:
    # Generated audio is not good enough for reliable pitch tracking
    return None
  else:
    keep_mask = ground_truth_f0_confidence >= f0_confidence_thresh

    delta_f0 = l1_distance(gen_f0, ground_truth_f0)
    delta_f0_filt = delta_f0[keep_mask]
    delta_f0_mean = np.mean(delta_f0_filt)
    # Report mean error in midi space for easier interpretation
    delta_f0_mean = delta_f0_mean * MAX_MIDI_PITCH

  return delta_f0_mean


# ---------------------- WAV files --------------------------------------------
def get_wav_file(wav_data, sample_rate):
  mem_file = io.BytesIO()
  try:
    wavfile.write(mem_file, sample_rate, wav_data)
  except:  # pylint: disable=bare-except
    logging.warning('error in writing WAV file')

  return mem_file


# ---------------------- Spectrogram ------------------------------------------
def spectrogram(audio, sess=None, rotate=False, size=2048):
  """Compute logmag spectrogram."""
  if sess is None:
    sess = tf.Session()
  mag = sess.run(
      ddsp.spectral_ops.calc_logmag(
          tf.convert_to_tensor(audio, tf.float32), size=size))
  if rotate:
    mag = np.rot90(mag)
  return mag


# ---------------------- Summary Writers --------------------------------------
class NSynthWriter(object):
  """Base Class for writing tensorboard summaries for RT Nsynth dataset."""

  def __init__(self, batch_size, summary_dir, global_step, verbose=True):
    """Initializes the result writer."""
    self._batch_size = batch_size
    self._summary_dir = summary_dir
    self._global_step = global_step
    self._file_writer = tf.summary.FileWriter(self._summary_dir)
    self._verbose = verbose

  def update(self, gen_audio_outputs, ground_truth_feats, tensor_dict):
    raise NotImplementedError('update() not defined')

  def flush(self):
    raise NotImplementedError('flush() not defined')


class NSynthWriters(object):
  """Result writer that wraps a list of NSynth writers."""

  def __init__(self, writers=None):
    """Initializes the result writer.

    Args:
      writers: list of `eval_utils.NSynthWriter`
    """
    self._writers = writers or []

  def add(self, writer):
    self._writers.append(writer)

  def update(self, gen_audio_outputs, ground_truth_feats, tensor_dict=None):
    for writer in self._writers:
      writer.update(gen_audio_outputs, ground_truth_feats, tensor_dict)

  def flush(self):
    for writer in self._writers:
      writer.flush()


class NSynthMetricsWriter(NSynthWriter):
  """Class for writing WaveRNN metrics in NSynth Dataset to tensorboard."""

  def __init__(self, batch_size, summary_dir, global_step):
    super(NSynthMetricsWriter, self).__init__(batch_size, summary_dir,
                                              global_step)
    self._metrics_dict = {
        'ld_metric': 0,
        'ld_dist_sum': 0,
        'ld_count': 0,
        'f0_metric': 0,
        'f0_outlier_ratio': 0,
        'f0_dist_sum': 0,
        'f0_ground_truth_untrackable_pitch_count': 0,
        'f0_gen_pitch_outlier_count': 0,
        'f0_gen_untrackable_pitch_count': 0,
        'f0_gen_trackable_pitch_count': 0,
    }

  def _compute_ld_dist_and_update_counts(self, gen_ld, ground_truth_ld):
    metrics_d = self._metrics_dict
    ld_dist = mean_l1_loudness(gen_ld, ground_truth_ld)
    metrics_d['ld_dist_sum'] += ld_dist
    metrics_d['ld_count'] += 1
    return ld_dist

  def _compute_f0_dist_and_update_counts(self, gen_f0, gen_f0_confidence,
                                         ground_truth_f0,
                                         ground_truth_f0_confidence):
    """Compute f0 dist and update corresponding counts."""
    metrics_d = self._metrics_dict
    if is_outlier(ground_truth_f0_confidence):
      # Ground truth f0 was unreliable to begin with. Discard.
      metrics_d['f0_ground_truth_untrackable_pitch_count'] += 1
      f0_dist = None
    else:
      # Gound truth f0 was reliable, compute f0 distance with generated audio
      f0_dist = f0_dist_conf_thresh(gen_f0, gen_f0_confidence, ground_truth_f0,
                                    ground_truth_f0_confidence)
      if f0_dist is not None and f0_dist > OUTLIER_MIDI_THRESH:
        # Generated audio had trackable pitch content but is an outlier
        metrics_d['f0_gen_pitch_outlier_count'] += 1
      elif f0_dist is not None and f0_dist <= OUTLIER_MIDI_THRESH:
        # Generated audio had trackable pitch content and is within tolerance
        metrics_d['f0_dist_sum'] += f0_dist
        metrics_d['f0_gen_trackable_pitch_count'] += 1
      elif f0_dist is None:
        # Generated audio had untrackable pitch content
        metrics_d['f0_gen_untrackable_pitch_count'] += 1
    return f0_dist

  def _compute_update_ld_metric(self):
    """Compute and update ld metric."""
    metrics_d = self._metrics_dict
    if metrics_d['ld_count'] == 0:
      ld_metric = np.nan
    else:
      ld_metric = metrics_d['ld_dist_sum'] / metrics_d['ld_count']
    metrics_d['ld_metric'] = ld_metric
    return ld_metric

  def _compute_update_f0_metric(self):
    """Compute and update f0 metric."""
    metrics_d = self._metrics_dict
    if metrics_d['f0_gen_trackable_pitch_count'] == 0:
      f0_metric = np.nan
    else:
      f0_metric = metrics_d['f0_dist_sum'] / metrics_d[
          'f0_gen_trackable_pitch_count']
    metrics_d['f0_metric'] = f0_metric
    return f0_metric

  def _compute_update_outlier_ratio(self):
    """Compute and update the outlier ratio.

    Outlier ratio distinguishes the number of poorly generated audio by the
    model vs.audio with poor pitch tracking to begin with.

    The lowest (best) possible ratio is `f0_ground_truth_untrackable_pitch_count
    / total_count` = 0.02. Indicating all generated samples were of good
    quality, and only ground truth samples with poor pitch content to begin with
    had to be omitted from evaluation.

    The outlier ratio is computed using:

    f0_ground_truth_untrackable_pitch_count +
    f0_gen_pitch_outlier_count +
    f0_gen_untrackable_pitch_count
                    /
    f0_ground_truth_untrackable_pitch_count +
    f0_gen_pitch_outlier_count +
    f0_gen_untrackable_pitch_count +
    f0_gen_trackable_pitch_count

    As the model improves in performance `f0_gen_pitch_outlier_count` and
    `f0_gen_untrackable_pitch_count` should decrease, causing a lower ratio.

    Args: None

    Returns:
      outlier_ratio: float or np.nan if division by 0
    """
    metrics_d = self._metrics_dict

    numerator = metrics_d['f0_ground_truth_untrackable_pitch_count']
    numerator += metrics_d['f0_gen_pitch_outlier_count']
    numerator += metrics_d['f0_gen_untrackable_pitch_count']

    denominator = copy.copy(numerator)
    denominator += metrics_d['f0_gen_trackable_pitch_count']

    if denominator == 0:
      outlier_ratio = np.nan
    else:
      outlier_ratio = numerator / denominator
    metrics_d['f0_outlier_ratio'] = outlier_ratio
    return outlier_ratio

  def update(self, gen_audio_outputs, ground_truth_feats, tensor_dict=None):
    """Update metrics dictionary given a batch of audio."""
    # Compute metrics per sample. No batch operations possible.
    for sample_idx in range(self._batch_size):
      # Extract generated audio
      gen_audio = check_and_squeeze_to_vector(
          gen_audio_outputs[sample_idx])
      gen_feats = compute_audio_features(gen_audio)

      ld_dist = self._compute_ld_dist_and_update_counts(
          gen_feats['loudness'], ground_truth_feats['loudness'][sample_idx])

      f0_dist = self._compute_f0_dist_and_update_counts(
          gen_feats['f0'], gen_feats['f0_confidence'],
          ground_truth_feats['f0'][sample_idx],
          ground_truth_feats['f0_confidence'][sample_idx])

      if self._verbose:
        log_string = 'sample {} | ld_dist: {:.3f} | '.format(
            sample_idx, ld_dist)
        if f0_dist:
          # Only log f0 distance if it was calculated
          log_string = log_string + 'f0_dist(midi): {:.3f}'.format(f0_dist)
        logging.info(log_string)

  def get_current_metrics(self):
    _ = self._compute_update_ld_metric()
    _ = self._compute_update_f0_metric()
    _ = self._compute_update_outlier_ratio()
    return self._metrics_dict

  def flush(self):
    """Output metrics to tensorboard for global step."""

    metrics_d = self.get_current_metrics()
    if self._verbose:
      logging.info('COMPUTING METRICS COMPLETE. FLUSHING ALL METRICS')

      metric_keys = ['ld_metric', 'f0_metric', 'f0_outlier_ratio']
      metrics_str = ' | '.join(
          '{}: {:0.3f}'.format(m, metrics_d[m]) for m in metric_keys)
      logging.info(metrics_str)

      counts_keys = [
          'f0_gen_trackable_pitch_count', 'f0_gen_pitch_outlier_count',
          'f0_gen_untrackable_pitch_count',
          'f0_ground_truth_untrackable_pitch_count'
      ]
      counts_str = ' | '.join(
          '{}: {}'.format(c, metrics_d[c]) for c in counts_keys)
      logging.info(counts_str)

    summary = tf.Summary()
    for value_name in [
        'ld_metric',
        'f0_metric',
        'f0_outlier_ratio',
    ]:
      summary.value.add(
          tag='rt_metrics/' + value_name, simple_value=metrics_d[value_name])

    for value_name in [
        'f0_gen_trackable_pitch_count', 'f0_gen_pitch_outlier_count',
        'f0_gen_untrackable_pitch_count',
        'f0_ground_truth_untrackable_pitch_count'
    ]:
      summary.value.add(
          tag='counts/' + value_name, simple_value=metrics_d[value_name])

    self._file_writer.add_summary(summary, self._global_step)

    self._file_writer.flush()
    logging.info('Wrote metric summaries for step %s to %s', self._global_step,
                 self._summary_dir)

    crepe.reset()  # Reset CREPE global state


class NSynthWaveformImageWriter(NSynthWriter):
  """Class for writing waveform tensorboard summaries."""

  def __init__(self, batch_size, summary_dir, global_step):
    super(NSynthWaveformImageWriter, self).__init__(batch_size, summary_dir,
                                                    global_step)

  def update(self, gen_audio_outputs, ground_truth_feats, tensor_dict):
    """Update metrics dictionary given a batch of audio."""
    gt = ground_truth_feats['audio']
    a1 = tensor_dict.get('additive_audio')
    a2 = tensor_dict.get('noise_audio')
    a3 = gen_audio_outputs

    def _plot(summary, sample_idx, length=None, prefix='waveform'):
      """Plots all waveforms."""
      waveforms = []
      labels = []
      for a, label in zip([gt, a1, a2, a3],
                          ['gt', 'additive', 'noise', 'synthesized']):
        if a is not None:
          x = check_and_squeeze_to_vector(a[sample_idx])
          if length is not None:
            x = x[:length]
          waveforms.append(x)
          labels.append(label)

      # Manually specify exact size of fig for tensorboard
      num_subplots = len(labels)
      fig = plt.figure(figsize=(2.5 * num_subplots, 10))

      for i in range(num_subplots):
        ax = fig.add_subplot(num_subplots, 1, i + 1)
        ax.plot(waveforms[i])
        ax.set_title(labels[i])

      # Format and save plot to image
      buf = io.BytesIO()
      fig.savefig(buf, format='png')
      image_summary = tf.Summary.Image(encoded_image_string=buf.getvalue())
      plt.close(fig)

      summary.value.add(
          tag='{}/{}'.format(prefix, sample_idx), image=image_summary)

    summary = tf.Summary()
    for sample_idx in range(self._batch_size):
      _plot(summary, sample_idx, length=None, prefix='waveform_4s')
      _plot(summary, sample_idx, length=2000, prefix='waveform_125ms')

    self._file_writer.add_summary(summary, self._global_step)

  def flush(self):
    """Output metrics to tensorboard for global step."""
    self._file_writer.flush()
    logging.info('Wrote image summaries for step %s to %s', self._global_step,
                 self._summary_dir)


class NSynthSpectrogramImageWriter(NSynthWriter):
  """Class for writing spectrogram tensorboard summaries."""

  def __init__(self, batch_size, summary_dir, global_step):
    super(NSynthSpectrogramImageWriter, self).__init__(batch_size, summary_dir,
                                                       global_step)

  def update(self,
             gen_audio_outputs,
             ground_truth_feats,
             unused_tensor_dict,
             sess=None):
    """Update metrics dictionary given a batch of audio."""
    # Batch spectrogram operations
    gtr_spectrograms = spectrogram(ground_truth_feats['audio'], sess=sess)
    gen_spectrograms = spectrogram(gen_audio_outputs, sess=sess)

    logging.info('spec writer')
    summary = tf.Summary()
    for sample_idx in range(self._batch_size):
      # Manually specify exact size of fig for tensorboard
      fig = plt.figure(figsize=(8, 5))

      ax = fig.add_subplot(111)
      ax.set_title('original')
      ax.set_xticks([0, 1000])
      ax.matshow(gtr_spectrograms[sample_idx], vmin=-5, vmax=1)

      ax = fig.add_subplot(212)
      ax.set_title('synthesized')
      ax.set_xticks([])
      ax.set_yticks([])
      ax.matshow(gen_spectrograms[sample_idx], vmin=-5, vmax=1)

      # Format and save plot to image
      buf = io.BytesIO()
      fig.savefig(buf, format='png')
      image_summary = tf.Summary.Image(encoded_image_string=buf.getvalue())
      plt.close(fig)

      summary.value.add(
          tag='spectrogram/{}'.format(sample_idx), image=image_summary)
    logging.info('spec writer 2')

    self._file_writer.add_summary(summary, self._global_step)

  def flush(self):
    """Output metrics to tensorboard for global step."""
    self._file_writer.flush()
    logging.info('Wrote image summaries for step %s to %s', self._global_step,
                 self._summary_dir)


class NSynthAudioWriter(NSynthWriter):
  """Class for writing audio samples to tensorboard."""

  def __init__(self, batch_size, summary_dir, global_step, sample_rate=16000):
    super(NSynthAudioWriter, self).__init__(batch_size, summary_dir,
                                            global_step)
    self._sample_rate = sample_rate

  def update(self, gen_audio_outputs, ground_truth_feats, unused_tensor_dict):
    """Update metrics dictionary given a batch of audio."""
    # Compute metrics per sample. No batch operations possible.
    summary = tf.Summary()
    for sample_idx in range(self._batch_size):
      # Ground truth audio
      gtr_audio = check_and_squeeze_to_vector(
          ground_truth_feats['audio'][sample_idx])

      gtr_audio_summary = tf.Summary.Audio(
          sample_rate=self._sample_rate,
          num_channels=1,
          length_frames=len(gtr_audio),
          encoded_audio_string=get_wav_file(gtr_audio,
                                            self._sample_rate).getvalue(),
          content_type='wav')

      summary.value.add(
          tag='ground_truth_audio/{}'.format(sample_idx),
          audio=gtr_audio_summary)

      # Synthesized audio
      gen_audio = check_and_squeeze_to_vector(
          gen_audio_outputs[sample_idx])

      gen_audio_summary = tf.Summary.Audio(
          sample_rate=self._sample_rate,
          num_channels=1,
          length_frames=len(gen_audio),
          encoded_audio_string=get_wav_file(gen_audio,
                                            self._sample_rate).getvalue(),
          content_type='wav')

      summary.value.add(
          tag='gen_audio/{}'.format(sample_idx), audio=gen_audio_summary)

    self._file_writer.add_summary(summary, self._global_step)

  def flush(self):
    """Output metrics to tensorboard for global step."""
    self._file_writer.flush()
    logging.info('Wrote audio summaries for step %s to %s', self._global_step,
                 self._summary_dir)


# ---------------------- Evaluation --------------------------------------------
@gin.configurable
def evaluate_or_sample(data_provider=gin.REQUIRED,
                       model=gin.REQUIRED,
                       mode='eval',
                       model_dir='~/tmp/ddsp/training',
                       master='',
                       batch_size=32,
                       num_batches=50,
                       keys_to_fetch='additive_audio,noise_audio',
                       ckpt_delay_secs=0,
                       run_once=False):
  """Run evaluation loop.

  Args:
    data_provider: DataProvider instance.
    model: Model instance.
    mode: Whether to 'eval' with metrics or create 'sample' s.
    model_dir: Path to directory with checkpoints and summary events.
    master: Name of TensorFlow runtime to use.
    batch_size: Size of each eval/sample batch.
    num_batches: How many batches to eval from dataset. -1 denotes all batches.
    keys_to_fetch: Additional tensors to fetch from model outputs.
    ckpt_delay_secs: Time to wait when a new checkpoint was not detected.
    run_once: Only run evaluation or sampling once.
  """
  # Set up dataset.
  input_fn = data_provider.get_input_fn(shuffle=False, repeats=1)
  model_dir = os.path.expanduser(model_dir)
  params = {'batch_size': batch_size, 'model_dir': model_dir}
  dataset = input_fn(params)
  dataset = dataset.take(num_batches)
  dataset = dataset.make_one_shot_iterator()
  features_tf = dataset.get_next()[0]

  # Load model checkpoint
  predictions = model.get_outputs(features_tf, is_training=False)
  # additional tensors to fetch during eval
  tensor_dict_tf = {}
  for k in keys_to_fetch.split(','):
    v = predictions.get(k)
    if v is not None:
      tensor_dict_tf[k] = v

  trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  saver = tf.train.Saver(var_list=trainable_variables)

  # Sample continuously and load the newest checkpoint each time
  checkpoints_iterator = tf.train.checkpoints_iterator(
      model_dir, ckpt_delay_secs)

  for checkpoint in checkpoints_iterator:

    # Set up writers before calling Sess() so computations run in same sess
    base_summary_dir = os.path.join(model_dir, 'summaries')
    if not tf.gfile.IsDirectory(base_summary_dir):
      tf.gfile.MakeDirs(base_summary_dir)

    global_step = int(checkpoint.split('-')[-1])

    writers = []

    if mode == 'eval':
      writers.append(
          NSynthMetricsWriter(
              batch_size=batch_size,
              summary_dir=base_summary_dir,
              global_step=global_step))

    elif mode == 'sample':
      writers.append(
          NSynthSpectrogramImageWriter(
              batch_size=batch_size,
              summary_dir=base_summary_dir,
              global_step=global_step))

      writers.append(
          NSynthWaveformImageWriter(
              batch_size=batch_size,
              summary_dir=base_summary_dir,
              global_step=global_step))

      writers.append(
          NSynthAudioWriter(
              batch_size=batch_size,
              summary_dir=base_summary_dir,
              global_step=global_step))

    nsynth_writers = NSynthWriters(writers)

    # Setup session.
    sess = tf.Session(target=master)
    sess.run(tf.global_variables_initializer())
    start_time = time.time()
    saver.restore(sess, checkpoint)
    logging.info('Loading model took %.1f seconds', time.time() - start_time)

    # Iterate through dataset and make predictions
    for batch_idx in itertools.count(1, 1):
      try:
        start_time = time.time()
        logging.info('Predicting batch %d', batch_idx)
        audio_gen, ground_truth_feats, tensor_dict = sess.run(
            (predictions['audio_gen'], features_tf, tensor_dict_tf))
        logging.info('Prediction took %.1f seconds', time.time() - start_time)

        nsynth_writers.update(
            audio_gen, ground_truth_feats, tensor_dict=tensor_dict)
        logging.info('Batch index %i with size %i took %.1f seconds', batch_idx,
                     batch_size,
                     time.time() - start_time)

      except tf.errors.OutOfRangeError:
        logging.info('End of dataset.')
        break

    nsynth_writers.flush()

    if run_once:
      break


