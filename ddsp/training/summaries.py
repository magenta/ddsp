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

"""Library of tensorboard summary functions relevant to DDSP training."""

import io

import ddsp
from ddsp.core import tf_float32
from ddsp.training.plotting import pianoroll_plot_setup
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import note_seq
from note_seq import sequences_lib
import numpy as np
import tensorflow.compat.v2 as tf


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
    waveform = np.squeeze(audio[i])
    waveform = waveform[:length] if length is not None else waveform
    waveform_gen = np.squeeze(audio_gen[i])
    waveform_gen = waveform_gen[:length] if length is not None else waveform_gen
    # Manually specify exact size of fig for tensorboard
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(2.5, 2.5))
    ax0.plot(waveform)
    ax1.plot(waveform_gen)

    # Format and save plot to image
    name = name + '_' if name else ''
    tag = f'waveform/{name}{prefix}_{i+1}'
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


def _plt_spec(spec, ax, title):
  """Helper function to plot a spectrogram to an axis."""
  spec = np.rot90(spec)
  ax.matshow(spec, vmin=-5, vmax=1, aspect='auto', cmap=plt.cm.magma)
  ax.set_title(title)
  ax.set_xticks([])
  ax.set_yticks([])


def spectrogram_summary(audio, audio_gen, step, name='', tag='spectrogram'):
  """Writes a summary of spectrograms for a batch of images."""
  specgram = lambda a: ddsp.spectral_ops.compute_logmag(tf_float32(a), size=768)

  # Batch spectrogram operations
  spectrograms = specgram(audio)
  spectrograms_gen = specgram(audio_gen)

  batch_size = int(audio.shape[0])
  name = name + '_' if name else ''

  for i in range(batch_size):
    # Manually specify exact size of fig for tensorboard
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    _plt_spec(spectrograms[i], axs[0], 'original')
    _plt_spec(spectrograms_gen[i], axs[1], 'synthesized')

    # Format and save plot to image
    tag_i = f'{tag}/{name}{i+1}'
    fig_summary(tag_i, fig, step)


def audio_summary(audio, step, sample_rate=16000, name='audio'):
  """Update metrics dictionary given a batch of audio."""
  # Ensure there is a single channel dimension.
  batch_size = int(audio.shape[0])
  if len(audio.shape) == 2:
    audio = audio[:, :, tf.newaxis]
  tf.summary.audio(
      name, audio, sample_rate, step, max_outputs=batch_size, encoding='wav')


def f0_summary(f0_hz, f0_hz_predict, step, name='f0_midi', tag='f0_midi'):
  """Creates a plot comparison of ground truth f0_hz and predicted values."""
  batch_size = int(f0_hz.shape[0])

  # Resample predictions to match ground truth if they don't already.
  if f0_hz.shape[1] != f0_hz_predict.shape[1]:
    f0_hz_predict = ddsp.core.resample(f0_hz_predict, f0_hz.shape[1])

  for i in range(batch_size):
    f0_midi = ddsp.core.hz_to_midi(tf.squeeze(f0_hz[i]))
    f0_midi_predict = ddsp.core.hz_to_midi(tf.squeeze(f0_hz_predict[i]))

    # Manually specify exact size of fig for tensorboard
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.0, 2.0))
    ax0.plot(f0_midi)
    ax0.plot(f0_midi_predict)
    ax0.set_title('original vs. predicted')

    ax1.plot(f0_midi_predict)
    ax1.set_title('predicted')

    # Format and save plot to image
    tag = f'{tag}/{name}_{i + 1}'
    fig_summary(tag, fig, step)


def midi_summary(controls, step, name, frame_rate, notes_key):
  """Plots segmented midi with controls."""
  batch_size = controls['f0_hz'].shape[0]
  for i in range(batch_size):

    amps = controls['harmonic']['controls']['amplitudes'][i]
    f0_hz = ddsp.core.hz_to_midi(controls['f0_hz'][i])

    fig, ax = plt.subplots(2, 1, figsize=(6.0, 4.0))
    ax[0].semilogy(amps, label='controls')
    ax[0].set_title('Amps')
    ax[1].plot(f0_hz, label='controls')
    ax[1].set_title('f0')

    notes_f0 = np.zeros_like(f0_hz)
    notes_amps = np.zeros_like(amps)
    markers = []

    note_sequence = controls[notes_key][i]
    for note in note_sequence.notes:
      start_time = int(note.start_time * frame_rate)
      end_time = int(note.end_time * frame_rate)
      notes_f0[start_time:end_time] = note.pitch
      notes_amps[start_time:end_time] = note.velocity / 1e3
      markers.append(start_time)

    ax[0].plot(notes_amps, '-d', label='notes', markevery=markers)
    ax[0].legend()
    ax[1].plot(notes_f0, '-d', label='notes', markevery=markers)
    ax[1].legend()

    fig_summary(f'midi/{name}_{i + 1}', fig, step)


def _get_reasonable_f0_min_max(f0_midi,
                               max_spike=5.0,
                               min_midi_value=5.0,
                               pad=6.0):
  """Find the min and max for an f0 plot, ignoring spike glitches and low notes.

   This function finds the min and max of the f0 after two operations to omit
   values. The first does np.diff() to examine the difference between adjacent
   values in the f0 curve. Values in abs(diff) above `max_spikes` are omitted.
   The second operation excludes MIDI notes below a threshold as determined by
   `min_midi_values`. After those two operations the min and max are found, and
   a padding (`pad`) value is added to the max and subtracted from the min
   before returning.

  Args:
    f0_midi: f0 curve in MIDI space.
    max_spike: Max value between successive diff values that will be included
      in the final min/max calculation.
    min_midi_value: Any MIDI values below this number will not be included in
      the final min/max calulation.
    pad: Value that will be added to the max and subtracted from the min.

  Returns:
    min_, max_: Values for an f0 plot that enphasizes the parts we care about.

  """

  # Mask out the 'spikes' by thresholding the diff above a value.
  diff = np.diff(f0_midi)
  diff = np.insert(diff, 0, 0.0)
  diff_mask = np.ma.masked_outside(diff, -max_spike, max_spike)

  # Remove any notes below the min.
  f0_mask = np.ma.masked_less(f0_midi, min_midi_value)

  # Combine the two masked arrays
  comb_masks = np.ma.array(
      f0_midi,
      mask=np.logical_or(diff_mask.mask, f0_mask.mask)
  )

  # Comute min/max with the padding and return.
  min_ = np.floor(np.min(comb_masks) - pad)
  max_ = np.ceil(np.max(comb_masks) + pad)
  return min_, max_


def _midiae_f0_helper(q_pitch, f0_midi, curve, i, step, label, tag):
  """Helper function to plot F0 info with MIDI AE."""
  min_, max_ = _get_reasonable_f0_min_max(f0_midi)
  plt.close('all')
  fig, ax, sp = pianoroll_plot_setup(figsize=(6.0, 4.0))
  sp.set_ylabel('MIDI Note Value')

  ax.step(q_pitch, 'r', linewidth=1.0, label='q_pitch')
  ax.plot(f0_midi, 'dodgerblue', linewidth=1.5, label='input f0')
  ax.plot(curve, 'darkgreen', linewidth=1.25, label=label)

  ax.set_ylim(min_, max_)
  ax.yaxis.set_major_locator(MaxNLocator(integer=True))
  ax.legend()
  fig_summary(f'{tag}/ex_{i + 1}', fig, step)


def midiae_f0_summary(f0_hz, outputs, step):
  """Makes plots to inspect f0/pitch components of MidiAE.

  Args:
    f0_hz: The input f0 to the network.
    outputs: Output dictionary from the MidiAe net.
    step: The step that the optimizer is currently on.
  """
  batch_size = int(f0_hz.shape[0])
  for i in range(batch_size):

    f0_midi = ddsp.core.hz_to_midi(tf.squeeze(f0_hz[i]))
    q_pitch = np.squeeze(outputs['q_pitch'][i])
    f0_rec = np.squeeze(outputs['f0_midi_pred'][i])
    _midiae_f0_helper(q_pitch, f0_midi, f0_rec, i, step, 'rec_f0',
                      'midiae_decoder_pitch')

    if 'f0_midi_rec2' in outputs:
      f0_rec2 = np.squeeze(outputs['f0_midi_pred2'][i])
      _midiae_f0_helper(q_pitch, f0_midi, f0_rec2, i, step, 'rec_f0_2',
                        'midiae_decoder_pitch2')

    if 'pitch' in outputs:
      raw_pitch = np.squeeze(outputs['pitch'][i])
      _midiae_f0_helper(q_pitch, f0_midi, raw_pitch, i, step, 'z_pitch',
                        'midiae_encoder_pitch')


def _midiae_ld_helper(ld_input, ld_rec, curve, db_key, i, step, label, tag):
  """Helper function to plot loudness info with MIDI AE."""
  fig = plt.figure(figsize=(6.0, 4.0))

  plt.plot(ld_input, linewidth=1.5, label='input ld')
  plt.plot(ld_rec, 'g', linewidth=1.25, label='rec ld')
  plt.step(curve, 'r', linewidth=0.75, label=label)

  plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
  plt.legend()
  fig_summary(f'{tag}/{db_key}_{i + 1}', fig, step)


def midiae_ld_summary(ld_feat, outputs, step, db_key='loudness_db'):
  """Makes plots to inspect loudness/velocity components of MidiAE.

  Args:
    ld_feat: The input loudness feature to the network.
    outputs: Output dictionary from the MidiAe net.
    step: The step that the optimizer is currently on
    db_key: Name of the loudness key (power_db or loudness_db).
  """
  batch_size = int(ld_feat.shape[0])
  for i in range(batch_size):
    ld_input = np.squeeze(ld_feat[i])
    ld_rec = np.squeeze(outputs[f'{db_key}_rec'][i])
    vel_quant = np.squeeze(outputs['velocity_quant'][i])

    _midiae_ld_helper(ld_input, ld_rec, vel_quant, db_key, i, step,
                      'q_vel', 'midiae_decoder_ld')

    if f'{db_key}_rec2' in outputs:
      ld_rec2 = np.squeeze(outputs[f'{db_key}_rec2'][i])
      _midiae_ld_helper(ld_input, ld_rec2, vel_quant, db_key, i, step,
                        'q_vel', 'midiae_decoder_ld2')

    if 'velocity' in outputs:
      vel = np.squeeze(outputs['velocity'][i])

      _midiae_ld_helper(ld_input, ld_rec, vel, db_key, i, step,
                        'vel', 'midiae_encoder_ld')


def midiae_sp_summary(outputs, step):
  """Synth Params summaries."""

  batch_size = int(outputs['f0_hz'].shape[0])
  have_pred = 'amps_pred' in outputs
  height = 12 if have_pred else 4
  rows = 3 if have_pred else 1

  for i in range(batch_size):
    # Amplitudes ----------------------------
    amps = np.squeeze(outputs['amps'][i])
    fig, ax = plt.subplots(nrows=rows, ncols=1, figsize=(8, height))
    ax[0].plot(amps)
    ax[0].set_title('Amplitudes - synth_params')

    if have_pred:
      amps_pred = np.squeeze(outputs['amps_pred'][i])
      ax[1].plot(amps_pred)
      ax[1].set_title('Amplitudes - pred')

      amps_diff = amps - amps_pred
      ax[2].plot(amps_diff)
      ax[2].set_title('Amplitudes - diff')

      for ax in fig.axes:
        ax.label_outer()

    fig_summary(f'amplitudes/amplitudes_{i + 1}', fig, step)

    # Harmonic Distribution ------------------
    hd = np.log(np.squeeze(outputs['hd'][i]) + 1e-8)
    fig, ax = plt.subplots(nrows=rows, ncols=1, figsize=(8, height))
    im = ax[0].imshow(hd.T, aspect='auto', origin='lower')
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title('Harmonic Distribution (log) - synth_params')

    if have_pred:
      hd_pred = np.log(np.squeeze(outputs['hd_pred'][i]) + 1e-8)
      im = ax[1].imshow(hd_pred.T, aspect='auto', origin='lower')
      fig.colorbar(im, ax=ax[1])
      ax[1].set_title('Harmonic Distribution (log) - pred')

      hd_diff = hd - hd_pred
      im = ax[2].imshow(hd_diff.T, aspect='auto', origin='lower')
      fig.colorbar(im, ax=ax[2])
      ax[2].set_title('Harmonic Distribution (log) - diff')

      for ax in fig.axes:
        ax.label_outer()

    fig_summary(f'harmonic_dist/harmonic_dist_{i + 1}', fig, step)

    # Magnitudes ----------------------------
    noise = np.squeeze(outputs['noise'][i])
    fig, ax = plt.subplots(nrows=rows, ncols=1, figsize=(8, height))
    im = ax[0].imshow(noise.T, aspect='auto', origin='lower')
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title('Noise mags - synth_params')

    if have_pred:
      noise_pred = np.squeeze(outputs['noise_pred'][i])
      im = ax[1].imshow(noise_pred.T, aspect='auto', origin='lower')
      fig.colorbar(im, ax=ax[1])
      ax[1].set_title('Noise mags - pred')

      noise_diff = noise - noise_pred
      im = ax[2].imshow(noise_diff.T, aspect='auto', origin='lower')
      fig.colorbar(im, ax=ax[2])
      ax[2].set_title('Noise mags - diff')
      for ax in fig.axes:
        ax.label_outer()

    fig_summary(f'noise_mags/noise_mags_{i + 1}', fig, step)


def pianoroll_summary(batch, step, name, frame_rate, pred_key,
                      gt_key='note_active_velocities', ch=None,
                      threshold=0.0, tb_name='pianoroll'):
  """Plots ground truth pianoroll against predicted MIDI."""
  batch_size = batch[gt_key].shape[0]
  for i in range(batch_size):
    if ch is None:
      gt_pianoroll = batch[gt_key][i]
      pred_pianoroll = batch[pred_key][i]
    else:
      gt_pianoroll = batch[gt_key][i, ..., ch]
      pred_pianoroll = batch[pred_key][i, ..., ch]

    if isinstance(pred_pianoroll, note_seq.NoteSequence):
      pred_pianoroll = sequences_lib.sequence_to_pianoroll(
          pred_pianoroll,
          frames_per_second=frame_rate,
          min_pitch=note_seq.MIN_MIDI_PITCH,
          max_pitch=note_seq.MAX_MIDI_PITCH).active[:-1, :]
    img = np.zeros((gt_pianoroll.shape[1], gt_pianoroll.shape[0], 4))

    # All values in `rgb` should be 0.0 except the value at index `idx`
    gt_color = {'idx': 1, 'rgb': np.array([0.0, 1.0, 0.0])}  # green
    pred_color = {'idx': 2, 'rgb': np.array([0.0, 0.0, 1.0])}  # blue

    gt_pianoroll_t = np.transpose(gt_pianoroll)
    pred_pianoroll_t = np.transpose(pred_pianoroll)
    img[:, :, gt_color['idx']] = gt_pianoroll_t
    img[:, :, pred_color['idx']] = pred_pianoroll_t

    # this is the alpha channel:
    img[:, :, 3] = np.logical_or(gt_pianoroll_t > threshold,
                                 pred_pianoroll_t > threshold)

    # Determine the min & max y-values for plotting.
    gt_note_indices = np.argmax(gt_pianoroll, axis=1)
    pred_note_indices = np.argmax(pred_pianoroll, axis=1)
    all_note_indices = np.concatenate([gt_note_indices, pred_note_indices])

    if np.sum(np.nonzero(all_note_indices)) > 0:
      lower_limit = np.min(all_note_indices[np.nonzero(all_note_indices)])
      upper_limit = np.max(all_note_indices)
    else:
      lower_limit = 0
      upper_limit = 127

    # Make the figures and add them to the summary.
    fig, ax, _ = pianoroll_plot_setup(figsize=(6.0, 4.0),
                                      xlim=[0, img.shape[1]])
    ax.imshow(img, origin='lower', aspect='auto', interpolation='nearest')
    ax.set_ylim((max(lower_limit - 5, 0), min(upper_limit + 5, 127)))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    labels_and_colors = [
        ('GT MIDI', gt_color['rgb']),  # green
        ('Pred MIDI', pred_color['rgb']),  # blue
        ('Overlap', gt_color['rgb'] + pred_color['rgb'])  # cyan
    ]
    patches = [mpatches.Patch(label=l, color=c) for l, c in labels_and_colors]
    fig.legend(handles=patches)
    fig_summary(f'{tb_name}/{name}_{i + 1}', fig, step)


