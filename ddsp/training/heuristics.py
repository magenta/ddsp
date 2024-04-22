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

"""Library containing DDSP output -> MIDI heurstics."""

import ddsp
import gin
import note_seq
import numpy as np
import tensorflow.compat.v2 as tf

DDSP_DEFAULT_FRAME_RATE = 250


def get_active_frame_indices(piano_roll):
  """Create matrix of frame indices for active notes relative to onset."""
  active_frame_indices = np.zeros_like(piano_roll)
  for frame_i in range(1, active_frame_indices.shape[0]):
    prev_indices = active_frame_indices[frame_i - 1, :]
    active_notes = piano_roll[frame_i, :]
    active_frame_indices[frame_i, :] = (prev_indices + 1) * active_notes
  return active_frame_indices


def _unbatch(batch):
  """Splits a dictionary of batched tensors into a list of dictionaries."""
  unbatched = []
  for key, val in batch.items():
    if isinstance(val, (tf.Tensor, np.ndarray)):
      if not unbatched:
        unbatched = [{} for _ in range(val.shape[0])]
      assert val.shape[0] == len(
          unbatched), f'batch size mismatch: {val.shape[0]} vs {len(unbatched)}'
      for i in range(val.shape[0]):
        unbatched[i][key] = val[i]
    elif isinstance(val, dict):
      sub_batch = _unbatch(val)
      if not unbatched:
        unbatched = [{} for _ in sub_batch]
      for i in range(len(sub_batch)):
        unbatched[i][key] = sub_batch[i]
    elif val is None:
      continue
    else:
      raise Exception(f'unsupported value at {key}:{val} of type {type(val)}')
  return unbatched


@gin.configurable
def segment_notes_batch(binarize_f,
                        pick_f0_f,
                        pick_amps_f,
                        controls_batch,
                        frame_rate=DDSP_DEFAULT_FRAME_RATE):
  """A function to split a controls dict into discrete notes.

  Args:
    binarize_f: Returns a binary vector that is True when a note is on.
    pick_f0_f: Returns a single f0 for a vector of f0s of a single note.
    pick_amps_f: Returns a single amplitude for a vector of amplidutes of a
      single note.
    controls_batch: The controls as returned from model inference, packed into a
      batch.
    frame_rate: Frame rate for the notes found.

  Returns:
    A list of NoteSequence objects, one for each element in the input batch.

  """
  notes_batch = []
  for controls in _unbatch(controls_batch):
    notes_batch.append(
        segment_notes(
            binarize_f=binarize_f,
            pick_f0_f=pick_f0_f,
            pick_amps_f=pick_amps_f,
            controls=controls,
            frame_rate=frame_rate))
  return notes_batch


def window_array(array, sr, win_len, frame_step_ratio=0.75, ax=0):
  """Chop up an array into overlapping frame windows."""
  frame_length = int(sr * win_len)
  frame_step = int(sr * win_len * frame_step_ratio)
  pad_front = int(sr * win_len * (1 - frame_step_ratio))

  padded_a = np.concatenate([np.zeros_like(array)[:pad_front], array], axis=ax)
  return tf.signal.frame(
      padded_a,
      frame_length=frame_length,
      frame_step=frame_step,
      pad_end=True,
      axis=ax).numpy()


def segment_notes(binarize_f,
                  pick_f0_f,
                  pick_amps_f,
                  controls,
                  frame_rate=DDSP_DEFAULT_FRAME_RATE):
  """A function to split a controls dict into discrete notes.

  Args:
    binarize_f: Returns a binary vector that is True when a note is on.
    pick_f0_f: Returns a single f0 for a vector of f0s of a single note.
    pick_amps_f: Returns a single amplitude for a vector of amplidutes of a
      single note.
    controls: The controls as returned from model inference.
    frame_rate: Frame rate for the notes found.

  Returns:
    NoteSequence object with discretized note information.

  """

  sequence = note_seq.NoteSequence()

  def construct_note(curr_ind, duration):
    note_start = curr_ind - duration
    f0 = pick_f0_f(controls, start=note_start, stop=curr_ind)
    amplitude = pick_amps_f(controls, start=note_start, stop=curr_ind)  # pylint:disable=unused-variable
    note = sequence.notes.add()
    note.pitch = np.round(ddsp.core.hz_to_midi(f0)).astype(np.int32)
    note.start_time = note_start / frame_rate
    note.end_time = (note_start + duration) / frame_rate
    # TODO(rigeljs): convert amplitude to velocity and add to note.
    note.velocity = 127

  binary_sample = binarize_f(controls)
  has_been_on = 0

  for i, sample_i in enumerate(np.nditer(binary_sample)):
    if sample_i:
      has_been_on += 1
    elif has_been_on > 0:
      construct_note(i, has_been_on)
      has_been_on = 0
  if has_been_on > 0:
    construct_note(len(binary_sample), has_been_on)

  sequence.total_time = len(binary_sample) / frame_rate

  return sequence


### PICK_F0_F candidates ###


@gin.register
def mean_f0(controls, start, stop):
  f0_hz = controls['f0_hz']
  return np.mean(f0_hz[start:stop])


@gin.register
def median_f0(controls, start, stop):
  f0_hz = controls['f0_hz']
  return np.median(f0_hz[start:stop])


### PICK_AMPS_F candidates ###


@gin.register
def median_amps(controls, start, stop):
  amps = np.squeeze(controls['harmonic']['controls']['amplitudes'])
  return np.median(amps[start:stop])


### BINARIZE_FN candidates ###


def remove_short(is_on_vec, min_samples=20, glue_back=False):
  """Removes short notes and optionally reattaches them to the previous note."""
  has_been_on = 0
  prev_note_end = 0
  for i, is_on in enumerate(np.nditer(is_on_vec, flags=('refs_ok',))):
    if is_on:
      has_been_on += 1
    else:
      if has_been_on < min_samples:
        # set this "on" stretch to off
        if glue_back:
          is_on_vec[prev_note_end:i] = True
        else:
          is_on_vec[i - has_been_on:i] = False
      has_been_on = 0
      prev_note_end = i
  return is_on_vec


def pad_for_frame(vec, mode, frame_width, axis=0):
  """A helper function to pad vectors for input to tf.signal.frame.

  Each element in vec is the center of a frame if frame_step == 1 after padding.

  Args:
    vec: The vector to be padded along the first dimension.
    mode: Either 'front', 'center', or 'end'.
    frame_width: Width of frame to pad for.
    axis: Axis to pad.

  Returns:
    The padded vector of shape [vec.shape[0] + pad_size].

  Raises:
    ValueError: If 'mode' passed in is not 'front', 'center', or 'end'.
  """
  if mode == 'front':
    pad_width_arg = (frame_width - 1, 0)
  elif mode == 'center':
    # handles even and odd frame widths
    pad_width_arg = int(frame_width / 2), frame_width - int(frame_width / 2) - 1
  elif mode == 'end':
    pad_width_arg = (0, frame_width - 1)
  else:
    raise ValueError(f'Unrecognized pad mode {mode}.')
  return np.pad(
      vec,
      pad_width_arg,
      mode='constant',
      constant_values=(int(np.take(vec, 0, axis)), int(np.take(vec, -1, axis))))


@gin.register
def amp_pooled_outliers(controls,
                        frame_width=80,
                        num_devs=2,
                        pad_mode='center'):
  """Finds amps that are n std devs below the mean of their neighbors."""
  log_amps = np.log(np.squeeze(controls['harmonic']['controls']['amplitudes']))
  padded_amps = pad_for_frame(
      log_amps, mode=pad_mode, frame_width=frame_width, axis=0)
  frames = tf.signal.frame(padded_amps, frame_width, 1)
  low_pooled = np.mean(frames, axis=-1) - (num_devs * np.std(frames, axis=-1))

  return low_pooled < log_amps


@gin.register
def strided_freq_change(controls,
                        frame_widths=(2, 4, 8, 16, 32),
                        pad_mode='front'):
  """Finds changes in f0 of >= 1 semitone across multiple strides."""
  f0 = np.squeeze(controls['f0_hz'].numpy())
  f0_midi = ddsp.core.hz_to_midi(f0)
  transitions = np.ones(len(f0), dtype=bool)
  for frame_width in frame_widths:
    padded_f0 = pad_for_frame(
        f0_midi, mode=pad_mode, frame_width=frame_width, axis=0)
    frames = tf.signal.frame(padded_f0, frame_width, 1)
    semitone_changes = np.abs(frames[..., 0] - frames[..., -1]) > 0.75

    # only flip if other strides didn't find this transition
    padded_transitions = pad_for_frame(
        transitions, mode=pad_mode, frame_width=frame_width, axis=0)
    prev_transitions = tf.signal.frame(
        padded_transitions, frame_width, 1, axis=-1)
    boundary_indexer = np.argwhere(
        np.all(prev_transitions, axis=-1) & semitone_changes)
    transitions[boundary_indexer] = False

  return transitions & (f0 > 0)


@gin.register
def power_pooled_outliers(controls,
                          frame_width=80,
                          num_devs=2.5,
                          pad_mode='center'):
  """Finds loudnesses that are n std devs below the mean of their neighbors."""
  shifted_power = ddsp.spectral_ops.compute_power(
      controls['audio'], frame_size=256) + ddsp.spectral_ops.LD_RANGE
  padded_power = pad_for_frame(
      shifted_power, mode=pad_mode, frame_width=frame_width, axis=0)
  frames = tf.signal.frame(padded_power, frame_width, 1)
  low_pooled = np.mean(frames, axis=-1) - (num_devs * np.std(frames, axis=-1))
  return np.asarray((low_pooled < shifted_power) & (shifted_power > 0))


@gin.register
def midi_heuristic(controls):
  """A combination of heuristics for extracting notes from DDSP controls."""
  return remove_short(
      strided_freq_change(controls) & amp_pooled_outliers(controls),
      min_samples=10)


@gin.register
def midi_heuristic_power(controls):
  """Like midi_heuristic, but uses power instead of extracted amplitudes."""
  return remove_short(
      strided_freq_change(controls) & power_pooled_outliers(controls),
      min_samples=10)
