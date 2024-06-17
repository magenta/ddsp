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

"""Library of performance metrics relevant to DDSP training."""

import dataclasses

from absl import logging
import ddsp
import librosa
import mir_eval
import note_seq
from note_seq import sequences_lib
import numpy as np
import tensorflow.compat.v2 as tf

# Global values for evaluation.
MIN_F0_CONFIDENCE = 0.85
OUTLIER_MIDI_THRESH = 12

# pytype: disable=signature-mismatch  # overriding-parameter-count-checks


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


def compute_audio_features(audio, frame_rate=250):
  """Compute features from audio."""
  audio_feats = {'audio': audio}
  audio = squeeze(audio)

  # Requires 16kHz for CREPE.
  sample_rate = ddsp.spectral_ops.CREPE_SAMPLE_RATE
  audio_feats['loudness_db'] = ddsp.spectral_ops.compute_loudness(
      audio, sample_rate, frame_rate)

  audio_feats['f0_hz'], audio_feats['f0_confidence'] = (
      ddsp.spectral_ops.compute_f0(audio, frame_rate))

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
  the original audio exceeds a minimum threshold.
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
class BaseMetrics(object):
  """Base object for computing metrics on generated audio samples."""

  def __init__(self, sample_rate, frame_rate, name):
    """Constructor.

    Args:
      sample_rate: Audio sample rate.
      frame_rate: Feature frame rate.
      name: Name to be printed when logging and scope for summaries.
    """
    self._sample_rate = sample_rate
    self._frame_rate = frame_rate
    self._name = name

  @property
  def metrics(self):
    """Initialize metrics dictionary with keys and keras metric objects."""
    raise NotImplementedError()

  def update_state(self):
    """Update running state of metric."""
    raise NotImplementedError()

  def flush(self, step):
    """Add summaries for each metric and reset the state."""
    # Log metrics
    logging.info('Computing %s metrics complete. Flushing all metrics',
                 self._name)
    metrics_str = ' | '.join(
        '{}: {:0.3f}'.format(k, v.result()) for k, v in self.metrics.items())
    logging.info(metrics_str)

    # Write tf summaries.
    for metric_name, metric in self.metrics.items():
      tf.summary.scalar('metrics/{}/{}'.format(self._name, metric_name),
                        metric.result(), step)
      metric.reset_states()


class LoudnessMetrics(BaseMetrics):
  """Helper object for computing loudness metrics."""

  def __init__(self, sample_rate, frame_rate, name='loudness'):
    super().__init__(sample_rate=sample_rate, frame_rate=frame_rate, name=name)
    self._metrics = {
        'loudness_db': tf.keras.metrics.Mean('loudness_db'),
    }

  @property
  def metrics(self):
    return self._metrics

  def update_state(self, batch, audio_gen):
    """Update metrics based on a batch of audio.

    Args:
      batch: Dictionary of input features.
      audio_gen: Batch of generated audio.
    """
    if 'loudness_db' in batch:
      loudness_original = batch['loudness_db']
    else:
      loudness_original = ddsp.spectral_ops.compute_loudness(
          batch['audio'], sample_rate=self._sample_rate,
          frame_rate=self._frame_rate)

    # Compute loudness across entire batch
    loudness_gen = ddsp.spectral_ops.compute_loudness(
        audio_gen, sample_rate=self._sample_rate,
        frame_rate=self._frame_rate)

    batch_size = int(audio_gen.shape[0])
    for i in range(batch_size):
      ld_dist = np.mean(l1_distance(loudness_original[i], loudness_gen[i]))
      self.metrics['loudness_db'].update_state(ld_dist)
      log_str = f'{self._name} | sample {i} | ld_dist(db): {ld_dist:.3f}'
      logging.info(log_str)


class F0CrepeMetrics(BaseMetrics):
  """Helper object for computing CREPE-based f0 metrics.

  Note that batch operations are not possible when CREPE has viterbi argument
  set to True.
  """

  def __init__(self, sample_rate, frame_rate, name='f0_crepe'):
    super().__init__(sample_rate=sample_rate, frame_rate=frame_rate, name=name)
    self._metrics = {
        'f0_dist':
            tf.keras.metrics.Mean('f0_dist'),
        'outlier_ratio':
            tf.keras.metrics.Accuracy('outlier_ratio'),
    }

  @property
  def metrics(self):
    return self._metrics

  def update_state(self, batch, audio_gen):
    """Update metrics based on a batch of audio.

    Args:
      batch: Dictionary of input features.
      audio_gen: Batch of generated audio.
    """
    batch_size = int(audio_gen.shape[0])
    # Compute metrics per sample. No batch operations possible.
    for i in range(batch_size):
      # Extract f0 from generated audio example.
      f0_hz_gen, _ = ddsp.spectral_ops.compute_f0(
          audio_gen[i],
          frame_rate=self._frame_rate,
          viterbi=True)
      if 'f0_hz' in batch and 'f0_confidence' in batch:
        f0_hz_gt = batch['f0_hz'][i]
        f0_conf_gt = batch['f0_confidence'][i]
      else:
        # Missing f0 in ground truth, extract it.
        f0_hz_gt, f0_conf_gt = ddsp.spectral_ops.compute_f0(
            batch['audio'][i],
            frame_rate=self._frame_rate,
            viterbi=True)

      if is_outlier(f0_conf_gt):
        # Ground truth f0 was unreliable to begin with. Discard.
        f0_dist = None
      else:
        # Gound truth f0 was reliable, compute f0 distance with generated audio
        f0_dist = f0_dist_conf_thresh(f0_hz_gt, f0_hz_gen, f0_conf_gt)
        if f0_dist is None or f0_dist > OUTLIER_MIDI_THRESH:
          # Generated audio had untrackable pitch content or is an outlier.
          self.metrics['outlier_ratio'].update_state(True, True)
          logging.info('sample %d has untrackable pitch content', i)
        else:
          # Generated audio had trackable pitch content and is within tolerance
          self.metrics['f0_dist'].update_state(f0_dist)
          self.metrics['outlier_ratio'].update_state(True, False)
          log_str = f'{self._name} | sample {i} | f0_dist(midi): {f0_dist:.3f}'
          logging.info(log_str)

  def flush(self, step):
    """Perform additional step of resetting CREPE."""
    super().flush(step)
    ddsp.spectral_ops.reset_crepe()  # Reset CREPE global state


class F0Metrics(BaseMetrics):
  """Helper object for computing f0 encoder metrics."""

  def __init__(
      self, sample_rate, frame_rate, rpa_tolerance=50, name='f0'):
    super().__init__(sample_rate=sample_rate, frame_rate=frame_rate, name=name)
    self._metrics = {
        'f0_dist': tf.keras.metrics.Mean('f0_dist'),
        'raw_pitch_accuracy': tf.keras.metrics.Mean('raw_chroma_accuracy'),
        'raw_chroma_accuracy': tf.keras.metrics.Mean('raw_pitch_accuracy')
    }
    self._rpa_tolerance = rpa_tolerance

  @property
  def metrics(self):
    return self._metrics

  def update_state(self, batch, f0_hz_predict):
    """Update metrics based on a batch of audio.

    Args:
      batch: Dictionary of input features.
      f0_hz_predict: Batch of encoded f0, same as input f0 if no f0 encoder.
    """
    batch_size = int(f0_hz_predict.shape[0])
    # Match number of timesteps.
    if f0_hz_predict.shape[1] != batch['f0_hz'].shape[1]:
      # f0_hz_predict = ddsp.core.resample(f0_hz_predict,
      #                                    batch['f0_hz'].shape[1]).numpy()
      batch['f0_hz'] = ddsp.core.resample(batch['f0_hz'],
                                          f0_hz_predict.shape[1]).numpy()
      batch['f0_confidence'] = ddsp.core.resample(
          batch['f0_confidence'], f0_hz_predict.shape[1]).numpy()

    # Compute metrics per sample. No batch operations possible.
    for i in range(batch_size):
      f0_hz_gt = batch['f0_hz'][i]
      f0_conf_gt = batch['f0_confidence'][i]

      if not is_outlier(f0_conf_gt):
        # Gound truth f0 was reliable, proceed with metrics
        # Compute distance between original f0_hz labels and f0 encoder values.
        # Resample if f0 encoder has different number of time steps.
        # TODO(hanoih): compare f0_hz against frame_rate * len_sec
        f0_hz = f0_hz_predict[i]
        f0_dist = f0_dist_conf_thresh(f0_hz_gt, f0_hz, f0_conf_gt)
        self.metrics['f0_dist'].update_state(f0_dist)

        f0_hz_gt = np.squeeze(f0_hz_gt)
        f0_hz = np.squeeze(f0_hz)
        voiced_gt = mir_eval.melody.freq_to_voicing(f0_hz_gt)[1]
        cents_gt = mir_eval.melody.hz2cents(f0_hz_gt)
        cents_est = mir_eval.melody.hz2cents(f0_hz)
        rca = mir_eval.melody.raw_chroma_accuracy(
            voiced_gt,
            cents_gt,
            voiced_gt,
            cents_est,
            cent_tolerance=self._rpa_tolerance)
        rpa = mir_eval.melody.raw_pitch_accuracy(
            voiced_gt,
            cents_gt,
            voiced_gt,
            cents_est,
            cent_tolerance=self._rpa_tolerance)
        self.metrics['raw_chroma_accuracy'].update_state(rca)
        self.metrics['raw_pitch_accuracy'].update_state(rpa)
        log_str = (f'{self._name} | sample {i} | f0_dist(midi): {f0_dist:.3f} '
                   f'raw_chroma_accuracy: {rca:.3f} '
                   f'raw_pitch_accuracy: {rpa:.3f}')
        logging.info(log_str)


@dataclasses.dataclass
class EvalCounts:
  """Data class to aggregate tp, fp and fn counts."""
  tp: int
  fp: int
  fn: int

  def add(self, other):
    self.tp += other.tp
    self.fp += other.fp
    self.fn += other.fn

  @property
  def precision(self):
    return self.tp / max((self.tp + self.fp), 1)

  @property
  def recall(self):
    return self.tp / max((self.tp + self.fn), 1)

  @property
  def f1(self):
    return (2 * self.precision * self.recall) / max(
        self.precision + self.recall, 1)


def sequence2intervals(sequence):
  """Convert a NoteSequence to a list of intervals for mir_eval."""
  notes = sequence.notes
  intervals = np.zeros((len(notes), 2))
  pitches = np.zeros(len(notes))
  velocities = np.zeros(len(notes))

  for i, note in enumerate(notes):
    intervals[i, 0] = note.start_time
    intervals[i, 1] = note.end_time
    pitches[i] = note.pitch
    velocities[i] = note.velocity

  return intervals, pitches, velocities


def compute_note_metrics(gt_sequence, pred_sequence):
  """Computes metrics for predicted sequence given ground truth in controls."""
  gt_intervals, gt_pitches, _ = sequence2intervals(gt_sequence)
  pred_intervals, pred_pitches, _ = sequence2intervals(pred_sequence)
  onset_matching = (
      mir_eval.transcription.match_notes(
          gt_intervals,
          ddsp.core.midi_to_hz(gt_pitches),
          pred_intervals,
          ddsp.core.midi_to_hz(pred_pitches),
          offset_ratio=None))
  onset_metrics = EvalCounts(
      tp=len(onset_matching),
      fp=len(pred_pitches) - len(onset_matching),
      fn=len(gt_pitches) - len(onset_matching))

  offset_matching = (
      mir_eval.transcription.match_notes(gt_intervals,
                                         ddsp.core.midi_to_hz(gt_pitches),
                                         pred_intervals,
                                         ddsp.core.midi_to_hz(pred_pitches)))

  full_note_metrics = EvalCounts(
      tp=len(offset_matching),
      fp=len(pred_pitches) - len(offset_matching),
      fn=len(gt_pitches) - len(offset_matching))

  return onset_metrics, full_note_metrics


def compute_frame_metrics(gt_pianoroll, pred_pianoroll):
  """Counts TP/FP/FN for framewise note activity assuming single-note audio."""
  gt_arr = np.squeeze(gt_pianoroll).max(axis=1)
  pred_arr = np.squeeze(pred_pianoroll).max(axis=1)
  assert gt_arr.shape == pred_arr.shape
  tp = np.logical_and(pred_arr > 0, gt_arr > 0).sum()
  fp = np.logical_and(pred_arr > 0, gt_arr == 0).sum()
  fn = np.logical_and(pred_arr == 0, gt_arr > 0).sum()
  return EvalCounts(tp, fp, fn)


class MidiMetrics(object):
  """A helper class to compute metrics for MIDI output."""

  def __init__(self,
               frames_per_second,
               tag,
               align_midi_with_f0=True):
    self.tag = tag
    self.note_counts = EvalCounts(0, 0, 0)
    self.note_offset_counts = EvalCounts(0, 0, 0)
    self.frame_counts = EvalCounts(0, 0, 0)
    self._frames_per_second = frames_per_second
    self._align_midi_with_f0 = align_midi_with_f0

  def _safe_convert_to_note_sequence(self, representation):
    """If the input is not a NoteSequence, convert it, else return it as is."""
    if isinstance(representation, note_seq.NoteSequence):
      return representation
    else:
      return sequences_lib.pianoroll_to_note_sequence(
          representation,
          frames_per_second=self._frames_per_second,
          min_duration_ms=0)

  def _safe_convert_to_pianoroll(self, representation):
    """If the input is not a piano roll, convert it, else return it as is."""
    pr_types = [sequences_lib.Pianoroll, np.ndarray, tf.Tensor]
    if any(isinstance(representation, t) for t in pr_types):
      return representation
    else:
      return sequences_lib.sequence_to_pianoroll(
          representation,
          frames_per_second=self._frames_per_second,
          min_pitch=note_seq.MIN_MIDI_PITCH,
          max_pitch=note_seq.MAX_MIDI_PITCH).active

  def align_midi_with_f0(self, gt_sequence, f0):
    """Align the notes in the NoteSequence with the extracted f0_hz."""
    sr = self._frames_per_second
    f0_midi = np.round(ddsp.core.hz_to_midi(f0))
    for i, note in enumerate(gt_sequence.notes):
      # look for the first match in [end of last note, end of curr note]
      lower_bound = int(gt_sequence.notes[i - 1].end_time * sr) if i > 0 else 0
      upper_bound = int(note.end_time * sr)
      for j in range(lower_bound, upper_bound):
        if int(f0_midi[j]) == note.pitch:
          note.start_time = float(j) / sr
          break

      # look for the last match in [start of curr note, start of next note]
      lower_bound = int(note.start_time * sr)
      upper_bound = int(
          gt_sequence.notes[i + 1].start_time *
          sr) if i < (len(gt_sequence.notes) - 1) else len(f0_midi) - 1
      for j in range(upper_bound, lower_bound, -1):
        if int(f0_midi[j]) == note.pitch:
          note.end_time = float(j) / sr
          break
    return gt_sequence

  def update_state(self, controls_batch, pred_seq_batch,
                   gt_key='note_active_velocities', ch=None):
    """Update metrics with given controls and notes."""

    gt_pianoroll_batch = controls_batch[gt_key]
    for i in range(len(pred_seq_batch)):
      if ch is None:
        pred_sequence = pred_seq_batch[i]
        gt_pianoroll = gt_pianoroll_batch[i]
      else:
        pred_sequence = pred_seq_batch[i, ..., ch]
        gt_pianoroll = gt_pianoroll_batch[i, ..., ch]

      # ------ Note metrics
      gt_sequence = self._safe_convert_to_note_sequence(gt_pianoroll)
      pred_sequence = self._safe_convert_to_note_sequence(pred_sequence)

      if self._align_midi_with_f0:
        f0 = controls_batch['f0_hz'][i]
        gt_sequence = self.align_midi_with_f0(gt_sequence, f0)

      note_counts_i, note_offset_counts_i = compute_note_metrics(
          gt_sequence, pred_sequence)
      self.note_counts.add(note_counts_i)
      self.note_offset_counts.add(note_offset_counts_i)

      # ------ Framewise metrics

      # converting to/from note_seq adds empty frames at the end. Remove them.
      gt_len = gt_pianoroll.shape[0]
      pred_pianoroll = self._safe_convert_to_pianoroll(pred_sequence)
      pred_pianoroll = pred_pianoroll[:gt_len, :]

      frame_counts_i = compute_frame_metrics(gt_pianoroll, pred_pianoroll)
      self.frame_counts.add(frame_counts_i)

  def flush(self, step):
    """Write summaries and reset state."""
    def write_summaries(counts, prefix):
      tf.summary.scalar(f'{prefix}/f1', counts.f1, step)
      tf.summary.scalar(f'{prefix}/precision', counts.precision, step)
      tf.summary.scalar(f'{prefix}/recall', counts.recall, step)
      metric_log = (f'{prefix}/f1: {counts.f1:0.3f} | '
                    f'{prefix}/precision: {counts.precision:0.3f} | '
                    f'{prefix}/recall: {counts.recall:0.3f}')
      logging.info(metric_log)

    write_summaries(self.note_counts, f'metrics/midi/{self.tag}/onset')
    write_summaries(self.note_offset_counts,
                    f'metrics/midi/{self.tag}/full_note')
    write_summaries(self.frame_counts, f'metrics/midi/{self.tag}/frame')

    self.full_note_counts = EvalCounts(0, 0, 0)
    self.note_counts = EvalCounts(0, 0, 0)
    self.frame_counts = EvalCounts(0, 0, 0)


