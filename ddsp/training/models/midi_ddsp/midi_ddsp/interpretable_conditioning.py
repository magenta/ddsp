"""Calculate note expression controls from synthesis parameters."""

import tensorflow as tf
import ddsp
import ddsp.training
import math


def extract_harm_controls(synth_params, log_scale=True, stop_gradient=False):
  """Get harmonic synth controls from the outputs of the processor group."""
  f0 = synth_params['harmonic']['controls']['f0_hz']
  amps = synth_params['harmonic']['controls']['amplitudes']
  hd = synth_params['harmonic']['controls']['harmonic_distribution']
  noise = synth_params['filtered_noise']['controls']['magnitudes']

  if log_scale:
    amps = ddsp.spectral_ops.amplitude_to_db(amps, use_tf=True)
    noise = ddsp.spectral_ops.amplitude_to_db(noise, use_tf=True)

  if stop_gradient:
    amps = tf.stop_gradient(amps)
    hd = tf.stop_gradient(hd)
    noise = tf.stop_gradient(noise)

  return f0, amps, hd, noise


def get_pitch_deviation(f0_midi, f0, mask_large_diff=True):
  # f0_midi: [batch_size,T,1]
  f0_midi_midi_scale = ddsp.core.hz_to_midi(f0_midi)
  f0_midi_scale = ddsp.core.hz_to_midi(f0)
  pitch_deviation = f0_midi_midi_scale - f0_midi_scale
  if mask_large_diff:
    pitch_deviation = tf.where(tf.greater(tf.abs(pitch_deviation), 2.0), 0.0,
                               pitch_deviation)
  return pitch_deviation


def get_amplitude(amplitude):
  # amplitude: [batch_size,T,1]
  return tf.convert_to_tensor(amplitude)


def get_brightness(harmonic_distribution):
  # weighted average across harmonic_distribution
  # harmonic_distribution: [batch, nframe, nharmonics], already scaled and normalized
  nharmonics = harmonic_distribution.shape[-1]
  num_bin = tf.cast(tf.linspace(1, nharmonics, num=nharmonics), tf.float32)[
            tf.newaxis, tf.newaxis, :]
  return tf.reduce_mean(harmonic_distribution * num_bin, axis=-1, keepdims=True)


def get_noise_level(noise_magnitudes):
  # noise_magnitudes: [n_batch, n_frames, n_frequencies]
  noise_amp = tf.reduce_mean(noise_magnitudes, axis=-1, keepdims=True)
  return noise_amp


def get_normal_window(t, alpha=0.5):
  w = 0.5 * (1 - tf.math.cos(2 * math.pi * t / alpha))
  mask = tf.cast(tf.logical_and(t > 0.5 * alpha, t < (1 - 0.5 * alpha)),
                 tf.float32)
  w = tf.clip_by_value(w + mask, 0, 1)
  return w


def get_vibrato_feature(pitch_deviation,
                        note_mask,
                        sampling_interval=0.004,
                        min_note_length=50,
                        vibrato_rate_min=3,
                        vibrato_rate_max=9):
  batch_size = pitch_deviation.shape[0]
  total_length = pitch_deviation.shape[1]
  pitch_deviation_masked = note_mask * pitch_deviation
  pv_mean = ddsp.training.nn.pool_over_notes(pitch_deviation, note_mask,
                                             return_std=False)
  pitch_deviation_masked = note_mask * (
        pitch_deviation_masked - pv_mean)  # filter out DC component

  each_note_idx = tf.cumsum(note_mask, axis=1) * tf.cast(~(note_mask == 0),
                                                         tf.float32)
  each_note_len = tf.reduce_sum(note_mask, axis=1, keepdims=True)
  each_note_time_ratio = each_note_idx / each_note_len
  window = get_normal_window(each_note_time_ratio)
  pitch_deviation_masked *= window

  # [batch_size*max_regions, n_frames]
  pitch_deviation_masked = tf.reshape(
    tf.transpose(pitch_deviation_masked, [0, 2, 1]), [-1, total_length])

  T = sampling_interval
  N = pitch_deviation.shape[1]
  f = tf.linspace(0, int(1 / T), N)

  s_vibrato = tf.abs(tf.signal.rfft(pitch_deviation_masked))
  s_vibrato = tf.math.divide_no_nan(s_vibrato,
                                    tf.reshape(each_note_len, [-1, 1]))

  vibrato_rate_idx = tf.argmax(tf.cast(s_vibrato, tf.float32), axis=-1)

  vibrato_rate = tf.cast(tf.gather(f, vibrato_rate_idx), tf.float32)
  vibrato_extend = tf.gather_nd(s_vibrato, vibrato_rate_idx[:, tf.newaxis],
                                batch_dims=1)
  # replace nan caused by rfft zeros with 0
  vibrato_extend = tf.where(tf.math.is_nan(vibrato_extend), 0, vibrato_extend)

  # filter out vibrato between 3-9 hz
  vibrato_mask = tf.math.logical_and(vibrato_rate >= vibrato_rate_min,
                                     vibrato_rate <= vibrato_rate_max)
  # filter out vibrato extend > 0.012
  # vibrato_mask = tf.math.logical_and(vibrato_mask, vibrato_extend > 0.012)

  # note length > 50 frames
  vibrato_mask = tf.math.logical_and(vibrato_mask,
                                     tf.reshape(each_note_len, [-1]) >
                                     min_note_length)

  # vibrato more than one cycle
  more_than_one_cycle_mask = vibrato_rate > tf.math.divide_no_nan(
    1., tf.reshape(each_note_len, [-1]) * sampling_interval)
  vibrato_mask = tf.math.logical_and(vibrato_mask, more_than_one_cycle_mask)
  vibrato_mask = tf.cast(vibrato_mask, tf.float32)

  # construct output
  vibrato_extend = vibrato_mask * vibrato_extend
  vibrato_rate = vibrato_mask * vibrato_rate
  frame_wise_vibrato_rate = tf.reduce_sum(
    tf.reshape(vibrato_rate, [batch_size, 1, -1]) * note_mask, axis=-1,
    keepdims=True)
  frame_wise_vibrato_extend = tf.reduce_sum(
    tf.reshape(vibrato_extend, [batch_size, 1, -1]) * note_mask, axis=-1,
    keepdims=True)

  return frame_wise_vibrato_rate, frame_wise_vibrato_extend


def get_amplitudes_max_pos(amplitudes, note_mask):
  note_mask_reverse = tf.cast(tf.logical_not(tf.cast(note_mask, tf.bool)),
                              tf.float32)
  amplitudes_masked = note_mask * amplitudes + note_mask_reverse * -1000
  # multiply -1000 here preventing argmax to the mask
  each_note_idx = tf.cumsum(note_mask, axis=1) * tf.cast(~(note_mask == 0),
                                                         tf.float32)
  each_note_len = tf.reduce_max(each_note_idx, axis=1, keepdims=True)
  note_onset_index = tf.argmax(note_mask, axis=1)

  # index inside a note that achieves max amplitudes
  amplitudes_max_idx = tf.argmax(amplitudes_masked, axis=1) - note_onset_index
  amplitudes_max_pos = tf.math.divide_no_nan(
    tf.cast(amplitudes_max_idx[:, tf.newaxis, :], tf.float32), each_note_len)
  amplitudes_max_pos = tf.reduce_sum(amplitudes_max_pos * note_mask, axis=-1,
                                     keepdims=True)
  return amplitudes_max_pos


def get_attack_level(noise_level, note_mask):
  each_note_idx = tf.cumsum(note_mask, axis=1) * tf.cast(~(note_mask == 0),
                                                         tf.float32)
  attack_mask = tf.cast(tf.logical_and(each_note_idx > 0, each_note_idx <= 10),
                        tf.float32)  # pool over first 10 frames
  # [b, n, d]
  attack_notes_mean = ddsp.training.nn.get_note_moments(noise_level,
                                                        attack_mask,
                                                        return_std=False)
  # [b, t, n, d]
  attack_time_notes_mean = (attack_notes_mean[:, tf.newaxis, ...] *
                            note_mask[..., tf.newaxis])
  # [b, t, d]
  attack_level = tf.reduce_sum(attack_time_notes_mean, axis=2)
  return attack_level


def get_interpretable_conditioning(f0_midi, f0, amplitude,
                                   harmonic_distribution, noise_magnitudes):
  """Calculate conditioning from synthesis needed for
  calculating note expression controls."""
  pitch_deviation = get_pitch_deviation(f0_midi, f0)
  amplitude = get_amplitude(amplitude)
  brightness = get_brightness(harmonic_distribution)
  noise_level = get_noise_level(noise_magnitudes)
  conditioning = {
    'pitch_deviation': pitch_deviation,
    'amplitude': amplitude,
    'brightness': brightness,
    'noise_level': noise_level,
  }
  return conditioning


def adsr_get_note_mask(q_pitch, max_regions=200, note_on_only=True):
  """Get a binary mask for each note from a monophonic instrument based on
  ADSR model. This function is modified from ddsp.training.nn.get_note_mask.
  In addition to the note boundary, it individually pool the first (A)
  and last few frames (D+R) of a note.

  Each transition of the value creates a new region. Returns the mask of each
  region.
  Args:
    q_pitch: A quantized value, such as pitch or velocity. Shape
      [batch, n_timesteps] or [batch, n_timesteps, 1].
    max_regions: Maximum number of note regions to consider in the sequence.
      Also, the channel dimension of the output mask. Each value transition
      defines a new region, e.g. each note-on and note-off count as a separate
      region.
    note_on_only: Return a mask that is true only for regions where the pitch
      is greater than 0.

  Returns:
    A binary mask of each region [batch, n_timesteps, max_regions].
  """
  # Only batch and time dimensions.
  if len(q_pitch.shape) == 3:
    q_pitch = q_pitch[:, :, 0]

  # Get onset and offset points.
  edges = tf.abs(ddsp.spectral_ops.diff(q_pitch, axis=1)) > 0

  # Count endpoints as starts/ends of regions.
  edges = edges[:, :-1, ...]
  edges = tf.pad(edges,
                 [[0, 0], [1, 0]], mode='constant', constant_values=True)
  edges = tf.pad(edges,
                 [[0, 0], [0, 1]], mode='constant', constant_values=False)
  edges = tf.cast(edges, tf.int32)

  # Count up onset and offsets for each timestep.
  # Assumes each onset has a corresponding offset.
  # The -1 ensures that the 0th index is the first note.
  edge_idx = tf.cumsum(edges, axis=1) - 1

  # Create masks of shape [batch, n_timesteps, max_regions].
  note_mask = edge_idx[..., None] == tf.range(max_regions)[None, None, :]
  note_mask = tf.cast(note_mask, tf.float32)

  if note_on_only:
    # [batch, notes]
    note_pitches = ddsp.training.nn.get_note_moments(q_pitch, note_mask,
                                                     return_std=False)
    # [batch, time, notes]
    note_on = tf.cast(note_pitches > 0.0, tf.float32)[:, None, :]
    # [batch, time, notes]
    note_mask *= note_on
    note_on_time_dim = tf.reduce_sum(note_mask,
                                     axis=-1)  # note_on in time dimension

  # frame index for each note
  each_note_idx = tf.cumsum(note_mask, axis=1) * tf.cast(~(note_mask == 0),
                                                         tf.float32)
  each_note_idx_reverse = tf.cumsum(note_mask, axis=1, reverse=True) * tf.cast(
    ~(note_mask == 0), tf.float32)
  each_note_len = tf.reduce_max(each_note_idx, axis=1, keepdims=True) * tf.cast(
    each_note_idx > 0, tf.float32)

  each_note_idx_reduce = tf.reduce_sum(each_note_idx, axis=-1)
  each_note_idx_reverse_reduce = tf.reduce_sum(each_note_idx_reverse, axis=-1)
  each_note_len_reduce = tf.reduce_sum(each_note_len, axis=-1)

  attack_mask = tf.math.logical_and(each_note_idx_reduce == 10,
                                    each_note_len_reduce >= 50)
  decay_mask = tf.math.logical_and(each_note_idx_reverse_reduce == 10,
                                   each_note_len_reduce >= 50)

  edges_adsr = edges + tf.cast(attack_mask, tf.int32) + tf.cast(decay_mask,
                                                                tf.int32)
  edge_idx_adsr = tf.cumsum(edges_adsr, axis=1) - 1

  # Create masks of shape [batch, n_timesteps, max_regions].
  note_mask_adsr = edge_idx_adsr[..., None] == tf.range(max_regions)[None, None,
                                               :]
  note_mask_adsr = tf.cast(note_mask_adsr, tf.float32)
  if note_on_only:
    note_mask_adsr *= note_on_time_dim[..., tf.newaxis]

  return note_mask_adsr


def get_conditioning_dict(conditioning, q_pitch, onsets,
                          pool_type='note_pooling'):
  """Calculate note expression controls."""
  # conditioning: dict of conditioning
  if pool_type == 'note_pooling':
    note_mask = ddsp.training.nn.get_note_mask_from_onset(q_pitch, onsets)
  elif pool_type == 'adsr_note_pooling':
    note_mask = adsr_get_note_mask(q_pitch)

  amp_mean, amp_std = ddsp.training.nn.pool_over_notes(
    conditioning['amplitude'], note_mask, return_std=True)
  brightness = ddsp.training.nn.pool_over_notes(conditioning['brightness'],
                                                note_mask, return_std=False)
  attack_level = get_attack_level(conditioning['noise_level'], note_mask)
  vibrato_rate, vibrato_extend = get_vibrato_feature(
    conditioning['pitch_deviation'], note_mask)
  amplitudes_max_pos = get_amplitudes_max_pos(conditioning['amplitude'],
                                              note_mask)

  # scale conditioning so that most value are in [0, 1]
  # TODO: (yusongwu) enable automatic scaling
  amp_mean = tf.where(tf.equal(amp_mean, 0.0), 0.0, amp_mean / 60 + 1.5)
  amp_std *= (2.5 / 60)
  vibrato_extend *= 10
  brightness *= 5
  attack_level = tf.where(tf.equal(attack_level, 0.0), 0.0,
                          attack_level / 40 + 2.625)

  conditioning_dict = {
    'amplitude_mean': amp_mean,
    'amplitude_std': amp_std,
    'vibrato_extend': vibrato_extend,
    'brightness': brightness,
    'attack_level': attack_level,
    # 'vibrato_rate': vibrato_rate,
    'amplitudes_max_pos': amplitudes_max_pos,
  }
  return conditioning_dict
