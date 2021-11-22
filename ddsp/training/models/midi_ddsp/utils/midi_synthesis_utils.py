"""Utility functions for synthesizing MIDI."""

import ddsp.core
import pretty_midi
import numpy as np
import tensorflow as tf
import os
from data_handling.instrument_name_utils import INST_ABB_TO_ID_DICT
from utils.audio_io import save_wav
from utils.inference_utils import ensure_same_length, \
  expression_generator_output_to_conditioning_df, \
  conditioning_df_to_dict, conditioning_df_to_audio, \
  conditioning_df_to_midi_features, CONDITIONING_KEYS


# MIDI Synthesis

def time_to_frame(time_s, fs=250):
  """Convert time in second to frame"""
  return int(round(time_s * fs))


def note_list_to_sequence(note_list, fs=250, pitch_offset=0, speed_rate=1):
  """Convert list of note in pretty_midi to midi sequence for
  expression generator."""
  note_pitch = []
  note_length = []

  prev_off_frame = 1000000
  for note in note_list:
    on_frame = time_to_frame(note.start / speed_rate, fs)
    off_frame = time_to_frame(note.end / speed_rate, fs)
    if on_frame - prev_off_frame > 0:
      note_pitch.append(0)
      note_length.append(on_frame - prev_off_frame)
    note_pitch.append(note.pitch + pitch_offset)
    note_length.append(off_frame - on_frame)
    prev_off_frame = off_frame

  # add a rest to the end
  note_pitch.append(0)
  note_length.append(fs)

  # adding batch dimension
  note_pitch = tf.convert_to_tensor(np.array(note_pitch))[tf.newaxis, ...]
  note_length = tf.convert_to_tensor(np.array(note_length))[tf.newaxis, ...]

  note_length = tf.cast(note_length, tf.float32)[..., tf.newaxis] * 0.004

  note_sequence = {'note_pitch': note_pitch, 'note_length': note_length}
  return note_sequence


def mono_midi_to_note_sequence(midi_file, instrument_id, pitch_offset=0,
                               speed_rate=1):
  """Convert a mono MIDI file to note sequence for expression generator."""
  midi_data = pretty_midi.PrettyMIDI(midi_file)
  instrument = midi_data.instruments[0]
  if instrument.is_drum:
    raise ValueError('Cannot synthesize drum')
  note_sequence = note_list_to_sequence(instrument.notes, fs=250,
                                        pitch_offset=pitch_offset,
                                        speed_rate=speed_rate)
  note_sequence['instrument_id'] = instrument_id
  return note_sequence


# Macro dicts for ensembles.
STRING_SET = {
  'Soprano': 'vn',
  'Alto': 'vn',
  'Tenor': 'va',
  'Bass': 'vc',
}

WOODWIND_SET = {
  'Soprano': 'fl',
  'Alto': 'ob',
  'Tenor': 'cl',
  'Bass': 'bn',
}

BRASSWIND_SET = {
  'Soprano': 'tpt',
  'Alto': 'hn',
  'Tenor': 'tbn',
  'Bass': 'tba',
}

MIX_SET = {
  'Soprano': 'cl',
  'Alto': 'tpt',
  'Tenor': 'sax',
  'Bass': 'tba',
}

QUARTET_SET_DICT = {
  'string_set': STRING_SET,
  'woodwind_set': WOODWIND_SET,
  'brasswind_set': BRASSWIND_SET,
  'mix_set': MIX_SET
}


def bach_midi_to_features(midi_file, quartet_set='woodwind_set',
                          quartet_set_dict=None, pitch_offset=0,
                          speed_rate=1):
  """Convert Bach chorales MIDI file to features for expression generator."""
  if quartet_set_dict is None:
    quartet_set_dict = QUARTET_SET_DICT

  midi_data = pretty_midi.PrettyMIDI(midi_file)
  note_sequence_all = []
  instrument_id_all = []
  part_name_all = []
  for instrument in midi_data.instruments:
    if instrument.is_drum:
      raise ValueError('Cannot synthesize drum')
    part_name = instrument.name.rstrip('\x00')
    note_sequence = note_list_to_sequence(instrument.notes, fs=250,
                                          pitch_offset=pitch_offset,
                                          speed_rate=speed_rate)
    instrument_abb = quartet_set_dict[quartet_set][part_name]
    instrument_id = tf.constant([INST_ABB_TO_ID_DICT[instrument_abb]])
    note_sequence['instrument_id'] = instrument_id
    instrument_id_all.append(instrument_id)
    note_sequence_all.append(note_sequence)
    part_name_all.append(part_name)
  return note_sequence_all, instrument_id_all, part_name_all


def synthesize_bach(synthesis_generator, expression_generator, midi_file,
                    quartet_set='string_set', pitch_offset=0,
                    speed_rate=1, output_dir=r'./', gain_adjust_db_dict={}):
  """Synthesize Bach chorales MIDI."""

  # Name for four parts.
  part_name_all = ['Soprano',
                   'Alto',
                   'Tenor',
                   'Bass', ]

  midi_data = pretty_midi.PrettyMIDI(midi_file)
  if len(midi_data.instruments) != 4:
    raise ValueError('The MIDI file do not have four parts')
  for i, instrument in enumerate(midi_data.instruments):
    instrument.name = part_name_all[i]
  midi_data.write(midi_file)

  # Create output directory.
  piece_name = os.path.basename(midi_file).replace('.mid', '')
  save_dir = os.path.join(output_dir, f'{piece_name}_{quartet_set}')
  os.makedirs(save_dir, exist_ok=True)

  # Synthesize the audio.
  note_sequence_all, instrument_id_all, part_name_all = bach_midi_to_features(
    midi_file, quartet_set=quartet_set,
    pitch_offset=pitch_offset,
    speed_rate=speed_rate)

  midi_audio_all = []
  conditioning_df_all = []
  for note_sequence, instrument_id in zip(note_sequence_all, instrument_id_all):
    expression_generator_outputs = expression_generator(note_sequence, out=None,
                                                        training=False)
    conditioning_df = expression_generator_output_to_conditioning_df(
      expression_generator_outputs['output'], note_sequence)
    conditioning_df_all.append(conditioning_df)

  midi_audio, midi_control_params, midi_synth_params = \
    batch_conditioning_df_to_audio(
      synthesis_generator,
      conditioning_df_all,
      instrument_id_all)

  # Save the stem audios.
  for i in range(midi_audio.shape[0]):
    midi_audio_all.append(midi_audio[i].numpy())
    save_wav(midi_audio[i].numpy(),
             os.path.join(save_dir, f'{part_name_all[i]}.wav'), 16000)

  # Mix the stems.
  if not gain_adjust_db_dict:
    gain_adjust_db_dict = {part_name: 0 for part_name in part_name_all}

  for i in range(len(midi_audio_all)):
    gain_adjust_amp_dict = {k: 10 ** (v / 20) for k, v in
                            gain_adjust_db_dict.items()}
    gain = gain_adjust_amp_dict[part_name_all[i]]
    midi_audio_all[i] = midi_audio_all[i] * gain

  # Save the mix.
  midi_audio_mix = tf.reduce_sum(tf.stack(midi_audio_all, axis=-1),
                                 axis=-1).numpy()
  save_wav(midi_audio_mix, os.path.join(save_dir, 'mix.wav'), 16000)

  return midi_audio_mix, midi_audio_all, midi_control_params, \
         midi_synth_params, conditioning_df_all


def batch_conditioning_df_to_audio(synthesis_generator, conditioning_df_all,
                                   instrument_id_all,
                                   display_progressbar=True):
  """Generate audio from a batch of conditioning_df."""
  conditioning_dict_all = [conditioning_df_to_dict(conditioning_df) for
                           conditioning_df in conditioning_df_all]
  conditioning_dict_all_concat = {}
  for key in conditioning_dict_all[0].keys():
    conditioning_dict_all_concat[key] = tf.concat(
      ensure_same_length([c[key] for c in conditioning_dict_all]),
      axis=0)
  conditioning_dict = conditioning_dict_all_concat

  midi_features_all = [conditioning_df_to_midi_features(conditioning_df) for
                       conditioning_df in conditioning_df_all]
  midi_features_all_concat = []
  for i in range(len(midi_features_all[0])):
    midi_features_all_concat.append(
      tf.concat(ensure_same_length([m[i] for m in midi_features_all]), axis=0))

  midi_features = tuple(midi_features_all_concat)
  instrument_id = tf.concat(instrument_id_all, axis=0)

  midi_audio, midi_control_params, midi_synth_params = synthesis_generator. \
    gen_audio_from_cond_dict(conditioning_dict,
                             midi_features,
                             instrument_id=instrument_id,
                             display_progressbar=display_progressbar)
  return midi_audio, midi_control_params, midi_synth_params


def synthesize_mono_midi(synthesis_generator, expression_generator, midi_file,
                         instrument_id, output_dir,
                         pitch_offset=0,
                         speed_rate=1,
                         display_progressbar=True):
  """Synthesize monophonic MIDI file.
  If MIDI file contains more than one part, will synthesize the first part."""
  note_sequence = mono_midi_to_note_sequence(midi_file,
                                             tf.constant([instrument_id]),
                                             pitch_offset=pitch_offset,
                                             speed_rate=speed_rate)
  expression_generator_outputs = expression_generator(note_sequence, out=None,
                                                      training=False)
  conditioning_df = expression_generator_output_to_conditioning_df(
    expression_generator_outputs['output'], note_sequence)
  midi_audio, midi_control_params, midi_synth_params = conditioning_df_to_audio(
    synthesis_generator, conditioning_df,
    tf.constant([instrument_id]),
    display_progressbar=display_progressbar)
  if output_dir is not None:
    save_wav(midi_audio[0].numpy(), os.path.join(output_dir, os.path.basename(
      midi_file).replace('.mid', '.wav')),
             16000)
  return midi_audio, midi_control_params, midi_synth_params, conditioning_df


def fill_conditioning_df_with_mean(conditioning_df, note_expression_stat,
                                   instrument_abb):
  """Fill conditioning_df with mean note expression values.
  Used for ablation test."""
  note_expression_list = CONDITIONING_KEYS
  for note_expression in note_expression_list:
    value = np.ones_like(conditioning_df[note_expression].values) * \
            note_expression_stat[instrument_abb][
              note_expression]
    value[conditioning_df['pitch'].values == 0] = 0
    if note_expression == 'vibrato_extend':
      value[conditioning_df['note_length'].values < 50] = 0
    conditioning_df[note_expression] = value
  return conditioning_df


def conditioning_df_to_f0_ld(conditioning_df, instrument_id, ld):
  """Convert a conditioning_df to f0 and loudness contour.
  Used for ablation test."""
  q_pitch, q_pitch, q_pitch, onsets, offsets = conditioning_df_to_midi_features(
    conditioning_df)
  f0 = ddsp.core.midi_to_hz(q_pitch, midi_zero_silence=True)
  ld = np.ones_like(f0) * ld
  ld[q_pitch == 0] = -120
  data = {
    'f0_hz': f0,
    'loudness_db': ld,
    'instrument_id': tf.constant([instrument_id])
  }
  return data


def conditioning_df_to_midi(conditioning_df, programs_id):
  """Convert a conditioning_df to a PrettyMIDI object."""
  midi_data = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(program=programs_id)
  frame_ms = 4
  for index, note in conditioning_df.iterrows():
    pitch = int(note['pitch'])
    if pitch != 0:
      on = int(note['onset'])
      off = int(note['offset'])
      n = pretty_midi.Note(velocity=100,
                           pitch=int(note['pitch']),
                           start=on * frame_ms / 1000,
                           end=off * frame_ms / 1000)
      instrument.notes.append(n)
  midi_data.instruments.append(instrument)
  return midi_data
