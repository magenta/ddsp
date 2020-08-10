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

import os
import time

from absl import logging
import ddsp
from ddsp.training import data
from ddsp.training import metrics
from ddsp.training import summaries
import gin
import numpy as np
import tensorflow.compat.v2 as tf


# ---------------------- Evaluation --------------------------------------------
def evaluate_or_sample(data_provider,
                       model,
                       mode='eval',
                       save_dir='~/tmp/ddsp/training',
                       restore_dir='',
                       batch_size=32,
                       num_batches=50,
                       ckpt_delay_secs=0,
                       run_once=False,
                       run_until_step=0):
  """Run evaluation loop.

  Args:
    data_provider: DataProvider instance.
    model: Model instance.
    mode: Whether to 'eval' with metrics or create 'sample' s.
    save_dir: Path to directory to save summary events.
    restore_dir: Path to directory with checkpoints, defaults to save_dir.
    batch_size: Size of each eval/sample batch.
    num_batches: How many batches to eval from dataset. -1 denotes all batches.
    ckpt_delay_secs: Time to wait when a new checkpoint was not detected.
    run_once: Only run evaluation or sampling once.
    run_until_step: Run until we see a checkpoint with a step greater or equal
      to the specified value. Ignored if <= 0.

  Returns:
    If the mode is 'eval', then returns a dictionary of Tensors keyed by loss
    type. Otherwise, returns None.
  """
  # Default to restoring from the save directory.
  restore_dir = save_dir if not restore_dir else restore_dir

  # Set up the summary writer and metrics.
  summary_dir = os.path.join(save_dir, 'summaries', 'eval')
  summary_writer = tf.summary.create_file_writer(summary_dir)

  # Sample continuously and load the newest checkpoint each time
  checkpoints_iterator = tf.train.checkpoints_iterator(restore_dir,
                                                       ckpt_delay_secs)

  # Get the dataset.
  dataset = data_provider.get_batch(batch_size=batch_size,
                                    shuffle=False,
                                    repeats=-1)

  # Get audio sample rate
  sample_rate = data_provider.sample_rate
  # Get feature frame rate
  frame_rate = data_provider.frame_rate

  latest_losses = None

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

          if isinstance(data_provider, data.SyntheticNotes):
            batch['audio'] = model.generate_synthetic_audio(batch)
            batch['f0_confidence'] = tf.ones_like(batch['f0_hz'])[:, :, 0]
            batch['loudness_db'] = ddsp.spectral_ops.compute_loudness(
                batch['audio'])

          elif isinstance(data_provider, data.ZippedProvider):
            batch, unused_ss_batch = model.parse_zipped_features(batch)

          # TODO(jesseengel): Find a way to add losses with training=False.
          audio = batch['audio']
          audio_gen, losses = model(batch, return_losses=True, training=True)
          audio_gen = np.array(audio_gen)
          outputs = model.get_controls(batch, training=True)

          # Create metrics on first batch.
          if mode == 'eval' and batch_idx == 1:
            loudness_metrics = metrics.LoudnessMetrics(
                sample_rate=sample_rate, frame_rate=frame_rate)
            f0_metrics = metrics.F0Metrics(
                sample_rate=sample_rate, frame_rate=frame_rate, name='f0_harm')
            f0_crepe_metrics = metrics.F0CrepeMetrics(
                sample_rate=sample_rate, frame_rate=frame_rate)

            f0_twm_metrics = metrics.F0Metrics(
                sample_rate=sample_rate, frame_rate=frame_rate, name='f0_twm')


            avg_losses = {
                name: tf.keras.metrics.Mean(name=name, dtype=tf.float32)
                for name in list(losses.keys())}

          for processor in model.processor_group.processors:
            # If using a sinusoidal model, infer f0 with two-way mismatch.
            if isinstance(processor, ddsp.synths.Sinusoidal):
              # Run on CPU to avoid running out of memory (not expensive comp).
              with tf.device('CPU'):
                processor_controls = outputs[processor.name]['controls']
                amps = processor_controls['amplitudes']
                freqs = processor_controls['frequencies']
                twm = ddsp.losses.TWMLoss()
                # Treat all freqs as candidate f0s.
                outputs['f0_hz_twm'] = twm.predict_f0(freqs, freqs, amps)
                logging.info('Added f0 estimate from sinusoids.')
                break

            # If using a noisy sinusoidal model, infer f0 with two-way mismatch.
            elif isinstance(processor, ddsp.synths.NoisySinusoidal):
              # Run on CPU to avoid running out of memory (not expensive comp).
              with tf.device('CPU'):
                processor_controls = outputs[processor.name]['controls']
                amps = processor_controls['amplitudes']
                freqs = processor_controls['frequencies']
                noise_ratios = processor_controls['noise_ratios']
                amps = amps * (1.0 - noise_ratios)
                twm = ddsp.losses.TWMLoss()
                # Treat all freqs as candidate f0s.
                outputs['f0_hz_twm'] = twm.predict_f0(freqs, freqs, amps)
                logging.info('Added f0 estimate from sinusoids.')
                break

          has_f0_twm = ('f0_hz_twm' in outputs and 'f0_hz' in batch)
          has_f0 = ('f0_hz' in outputs and 'f0_hz' in batch)

          logging.info('Prediction took %.1f seconds', time.time() - start_time)

          if mode == 'sample':
            start_time = time.time()
            logging.info('Writing summmaries for batch %d', batch_idx)

            # Add audio.
            summaries.audio_summary(
                audio_gen, step, sample_rate, name='audio_generated')
            summaries.audio_summary(
                audio, step, sample_rate, name='audio_original')

            # Add plots.
            summaries.waveform_summary(audio, audio_gen, step)
            summaries.spectrogram_summary(audio, audio_gen, step)
            if has_f0:
              summaries.f0_summary(batch['f0_hz'], outputs['f0_hz'], step,
                                   name='f0_harmonic')
            if has_f0_twm:
              summaries.f0_summary(batch['f0_hz'], outputs['f0_hz_twm'], step,
                                   name='f0_twm')

            logging.info('Writing batch %i with size %i took %.1f seconds',
                         batch_idx, batch_size, time.time() - start_time)

          elif mode == 'eval':
            start_time = time.time()
            logging.info('Calculating metrics for batch %d', batch_idx)

            loudness_metrics.update_state(batch, audio_gen)
            if has_f0:
              f0_metrics.update_state(batch, outputs['f0_hz'])
            else:
              f0_crepe_metrics.update_state(batch, audio_gen)
            if has_f0_twm:
              f0_twm_metrics.update_state(batch, outputs['f0_hz_twm'])
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
        loudness_metrics.flush(step)
        if has_f0:
          f0_metrics.flush(step)
        else:
          f0_crepe_metrics.flush(step)
        if has_f0_twm:
          f0_twm_metrics.flush(step)
        latest_losses = {}
        for k, metric in avg_losses.items():
          latest_losses[k] = metric.result()
          tf.summary.scalar('losses/{}'.format(k), metric.result(), step=step)
          metric.reset_states()

      summary_writer.flush()

      if run_once:
        break

      if 0 < run_until_step <= step:
        logging.info(
            'Saw checkpoint with step %d, which is greater or equal to'
            ' `run_until_step` of %d. Exiting.', step, run_until_step)
        break
  return latest_losses


@gin.configurable
def evaluate(data_provider,
             model,
             save_dir='~/tmp/ddsp/training',
             restore_dir='',
             batch_size=32,
             num_batches=50,
             ckpt_delay_secs=0,
             run_once=False,
             run_until_step=0):
  """Run evaluation loop.

  Args:
    data_provider: DataProvider instance.
    model: Model instance.
    save_dir: Path to directory to save summary events.
    restore_dir: Path to directory with checkpoints, defaults to save_dir.
    batch_size: Size of each eval/sample batch.
    num_batches: How many batches to eval from dataset. -1 denotes all batches.
    ckpt_delay_secs: Time to wait when a new checkpoint was not detected.
    run_once: Only run evaluation or sampling once.
    run_until_step: Run until we see a checkpoint with a step greater or equal
      to the specified value. Ignored if <= 0.

  Returns:
    A dictionary of tensors containing the loss values, keyed by loss type.

  """
  return evaluate_or_sample(
      data_provider=data_provider,
      model=model,
      mode='eval',
      save_dir=save_dir,
      restore_dir=restore_dir,
      batch_size=batch_size,
      num_batches=num_batches,
      ckpt_delay_secs=ckpt_delay_secs,
      run_once=run_once,
      run_until_step=run_until_step)


@gin.configurable
def sample(data_provider,
           model,
           save_dir='~/tmp/ddsp/training',
           restore_dir='',
           batch_size=16,
           num_batches=1,
           ckpt_delay_secs=0,
           run_once=False,
           run_until_step=0):
  """Run sampling loop.

  Args:
    data_provider: DataProvider instance.
    model: Model instance.
    save_dir: Path to directory to save summary events.
    restore_dir: Path to directory with checkpoints, defaults to save_dir.
    batch_size: Size of each eval/sample batch.
    num_batches: How many batches to eval from dataset. -1 denotes all batches.
    ckpt_delay_secs: Time to wait when a new checkpoint was not detected.
    run_once: Only run evaluation or sampling once.
    run_until_step: Run until we see a checkpoint with a step greater or equal
      to the specified value. Ignored if <= 0.
  """
  evaluate_or_sample(
      data_provider=data_provider,
      model=model,
      mode='sample',
      save_dir=save_dir,
      restore_dir=restore_dir,
      batch_size=batch_size,
      num_batches=num_batches,
      ckpt_delay_secs=ckpt_delay_secs,
      run_once=run_once,
      run_until_step=run_until_step)
