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
"""Library of performance metrics relevant to DDSP training."""

from absl import logging
import ddsp
import librosa
import numpy as np
import tensorflow.compat.v2 as tf

# Global values for evaluation.
MIN_F0_CONFIDENCE = 0.85
OUTLIER_MIDI_THRESH = 12



# ---------------------- Helper Functions --------------------------------------
def squeeze(input_vector):
  """Ensure vector only has one axis of dimensionality."""
  if input_vector.ndim > 1:
    return np.squeeze(input_vector)
  else:
    return input_vector


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
  if len(f0_hz.shape) > 2:
    f0_hz = f0_hz[:, :, 0]
  if len(f0_hz_gen.shape) > 2:
    f0_hz_gen = f0_hz_gen[:, :, 0]
  if len(f0_confidence.shape) > 2:
    f0_confidence = f0_confidence[:, :, 0]

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


# ---------------------- Metrics -----------------------------------------------
class F0LoudnessMetrics(object):
  """Helper object for computing f0 and loudness metrics."""

  def __init__(self, sample_rate):
    self.metrics = {
        'loudness_db': tf.keras.metrics.Mean('loudness_db'),
        'f0_encoder': tf.keras.metrics.Mean('f0_encoder'),
        'f0_crepe': tf.keras.metrics.Mean('f0_crepe'),
        'f0_crepe_outlier_ratio':
            tf.keras.metrics.Accuracy('f0_crepe_outlier_ratio'),
    }
    self._sample_rate = sample_rate

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
      feats_gen = compute_audio_features(
          audio_gen[i], sample_rate=self._sample_rate)

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
