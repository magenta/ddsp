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

"""Library of evaluator implementations for use in eval_util."""
import ddsp
from ddsp.training import heuristics
from ddsp.training import metrics
from ddsp.training import summaries
import gin
import numpy as np
import tensorflow.compat.v2 as tf


class BaseEvaluator(object):
  """Base class for evaluators."""

  def __init__(self, sample_rate, frame_rate):
    self._sample_rate = sample_rate
    self._frame_rate = frame_rate

  def set_rates(self, sample_rate, frame_rate):
    """Sets sample and frame rates, not known in gin initialization."""
    self._sample_rate = sample_rate
    self._frame_rate = frame_rate

  def evaluate(self, batch, output, losses):
    """Computes metrics."""
    raise NotImplementedError()

  def sample(self, batch, outputs, step):
    """Computes and logs samples."""
    raise NotImplementedError()

  def flush(self, step):
    """Logs metrics."""
    raise NotImplementedError()



@gin.register
class BasicEvaluator(BaseEvaluator):
  """Computes audio samples and losses."""

  def __init__(self, sample_rate, frame_rate):
    super().__init__(sample_rate, frame_rate)
    self._avg_losses = {}

  def evaluate(self, batch, outputs, losses):
    del outputs  # Unused.
    if not self._avg_losses:
      self._avg_losses = {
          name: tf.keras.metrics.Mean(name=name, dtype=tf.float32)
          for name in list(losses.keys())
      }
    # Loss.
    for k, v in losses.items():
      self._avg_losses[k].update_state(v)

  def sample(self, batch, outputs, step):
    audio = batch['audio']
    audio_gen = outputs['audio_gen']

    audio_gen = np.array(audio_gen)

    # Add audio.
    summaries.audio_summary(
        audio_gen, step, self._sample_rate, name='audio_generated')
    summaries.audio_summary(
        audio, step, self._sample_rate, name='audio_original')

    # Add plots.
    summaries.waveform_summary(audio, audio_gen, step)
    summaries.spectrogram_summary(audio, audio_gen, step)

  def flush(self, step):
    latest_losses = {}
    for k, metric in self._avg_losses.items():
      latest_losses[k] = metric.result()
      tf.summary.scalar('losses/{}'.format(k), metric.result(), step=step)
      metric.reset_states()


@gin.register
class F0LdEvaluator(BaseEvaluator):
  """Computes F0 and loudness metrics."""

  def __init__(self, sample_rate, frame_rate, run_f0_crepe=True):
    super().__init__(sample_rate, frame_rate)
    self._loudness_metrics = metrics.LoudnessMetrics(
        sample_rate=sample_rate, frame_rate=frame_rate)
    self._f0_metrics = metrics.F0Metrics(
        sample_rate=sample_rate, frame_rate=frame_rate)
    self._run_f0_crepe = run_f0_crepe
    if self._run_f0_crepe:
      self._f0_crepe_metrics = metrics.F0CrepeMetrics(
          sample_rate=sample_rate, frame_rate=frame_rate)

  def evaluate(self, batch, outputs, losses):
    del losses  # Unused.
    audio_gen = outputs['audio_gen']
    self._loudness_metrics.update_state(batch, audio_gen)

    if 'f0_hz' in outputs and 'f0_hz' in batch:
      self._f0_metrics.update_state(batch, outputs['f0_hz'])
    elif self._run_f0_crepe:
      self._f0_crepe_metrics.update_state(batch, audio_gen)

  def sample(self, batch, outputs, step):
    if 'f0_hz' in outputs and 'f0_hz' in batch:
      summaries.f0_summary(batch['f0_hz'], outputs['f0_hz'], step,
                           name='f0_harmonic')

  def flush(self, step):
    self._loudness_metrics.flush(step)
    self._f0_metrics.flush(step)
    if self._run_f0_crepe:
      self._f0_crepe_metrics.flush(step)


@gin.register
class TWMEvaluator(BaseEvaluator):
  """Evaluates F0s created with TWM heuristic."""

  def __init__(self,
               sample_rate,
               frame_rate,
               processor_name='sinusoidal',
               noisy=False):
    super().__init__(sample_rate, frame_rate)
    self._noisy = noisy
    self._processor_name = processor_name
    self._f0_twm_metrics = metrics.F0Metrics(
                sample_rate=sample_rate, frame_rate=frame_rate, name='f0_twm')

  def _compute_twm_f0(self, outputs):
    """Computes F0 from sinusoids using TWM heuristic."""
    processor_controls = outputs[self._processor_name]['controls']
    freqs = processor_controls['frequencies']
    amps = processor_controls['amplitudes']
    if self._noisy:
      noise_ratios = processor_controls['noise_ratios']
      amps = amps * (1.0 - noise_ratios)
    twm = ddsp.losses.TWMLoss()
    # Treat all freqs as candidate f0s.
    return twm.predict_f0(freqs, freqs, amps)

  def evaluate(self, batch, outputs, losses):
    del losses  # Unused.
    twm_f0 = self._compute_twm_f0(outputs)
    self._f0_twm_metrics.update_state(batch, twm_f0)

  def sample(self, batch, outputs, step):
    twm_f0 = self._compute_twm_f0(outputs)
    summaries.f0_summary(batch['f0_hz'], twm_f0, step, name='f0_twm')

  def flush(self, step):
    self._f0_twm_metrics.flush(step)


@gin.register
class MidiAutoencoderEvaluator(BaseEvaluator):
  """Metrics for MIDI Autoencoder."""

  def __init__(self, sample_rate, frame_rate, db_key='loudness_db',
               f0_key='f0_hz'):
    super().__init__(sample_rate, frame_rate)
    self._midi_metrics = metrics.MidiMetrics(
        frames_per_second=frame_rate, tag='learned')
    self._db_key = db_key
    self._f0_key = f0_key

  def evaluate(self, batch, outputs, losses):
    del losses  # Unused.
    self._midi_metrics.update_state(outputs, outputs['pianoroll'])

  def sample(self, batch, outputs, step):
    audio = batch['audio']
    summaries.audio_summary(
        audio, step, self._sample_rate, name='audio_original')

    audio_keys = ['midi_audio', 'synth_audio', 'midi_audio2', 'synth_audio2']
    for k in audio_keys:
      if k in outputs and outputs[k] is not None:
        summaries.audio_summary(outputs[k], step, self._sample_rate, name=k)
        summaries.spectrogram_summary(audio, outputs[k], step, tag=k)
        summaries.waveform_summary(audio, outputs[k], step, name=k)

    summaries.f0_summary(
        batch[self._f0_key], outputs[f'{self._f0_key}_pred'],
        step, name='f0_hz_rec')

    summaries.pianoroll_summary(outputs, step, 'pianoroll',
                                self._frame_rate, 'pianoroll')
    summaries.midiae_f0_summary(batch[self._f0_key], outputs, step)
    ld_rec = f'{self._db_key}_rec'
    if ld_rec in outputs:
      summaries.midiae_ld_summary(batch[self._db_key], outputs, step,
                                  self._db_key)

    summaries.midiae_sp_summary(outputs, step)

  def flush(self, step):
    self._midi_metrics.flush(step)


@gin.register
class MidiHeuristicEvaluator(BaseEvaluator):
  """Metrics for MIDI heuristic."""

  def __init__(self, sample_rate, frame_rate):
    super().__init__(sample_rate, frame_rate)
    self._midi_metrics = metrics.MidiMetrics(
        tag='heuristic', frames_per_second=frame_rate)

  def _compute_heuristic_notes(self, outputs):
    return heuristics.segment_notes_batch(
        binarize_f=heuristics.midi_heuristic,
        pick_f0_f=heuristics.mean_f0,
        pick_amps_f=heuristics.median_amps,
        controls_batch=outputs)

  def evaluate(self, batch, outputs, losses):
    del losses  # Unused.
    notes = self._compute_heuristic_notes(outputs)
    self._midi_metrics.update_state(outputs, notes)

  def sample(self, batch, outputs, step):
    notes = self._compute_heuristic_notes(outputs)
    outputs['heuristic_notes'] = notes
    summaries.midi_summary(outputs, step, 'heuristic', self._frame_rate,
                           'heuristic_notes')
    summaries.pianoroll_summary(outputs, step, 'heuristic',
                                self._frame_rate, 'heuristic_notes')

  def flush(self, step):
    self._midi_metrics.flush(step)


