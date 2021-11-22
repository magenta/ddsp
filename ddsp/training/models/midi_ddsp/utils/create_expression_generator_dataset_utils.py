"""Utility functions for creating dataset for expression generator dataset."""

import numpy as np
import ddsp
import tensorflow as tf
import copy
import glob
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from utils.file_utils import pickle_dump
from utils.inference_utils import to_length, make_same_length, get_process_group
from utils.audio_io import save_wav
from midi_ddsp.interpretable_conditioning import \
  get_interpretable_conditioning, get_amplitudes_max_pos, get_vibrato_feature, \
  extract_harm_controls
from data_handling.urmp_tfrecord_unbatched_dataloader import UrmpMidiUnbatched


# Sample the whole piece in URMP dataset.

def save_synth_params(data, synth_params, output_dir):
  recording_id = data['recording_id'][0].numpy().decode()
  synth_params = {k: v.numpy()[0] for k, v in synth_params.items()}
  synth_params_save_path = os.path.join(output_dir,
                                        f'{recording_id}.synth_params')
  np.save(synth_params_save_path, synth_params)
  midi_save_path = os.path.join(output_dir, f'{recording_id}.midi')
  np.save(midi_save_path, data['midi'].numpy()[0])
  onsets_save_path = os.path.join(output_dir, f'{recording_id}.onsets')
  np.save(onsets_save_path, data['onsets'].numpy()[0])
  offsets_save_path = os.path.join(output_dir, f'{recording_id}.offsets')
  np.save(offsets_save_path, data['offsets'].numpy()[0])
  instrument_id_save_path = os.path.join(output_dir,
                                         f'{recording_id}.instrument_id')
  np.save(instrument_id_save_path, data['instrument_id'].numpy()[0])
  wav_path = os.path.join(output_dir, f'{recording_id}.wav')
  np.save(wav_path, data['audio'].numpy()[0])
  save_wav(data['synth_audio'].numpy()[0],
           os.path.join(output_dir, f'{recording_id}_synth.wav'), 16000)
  assert len(synth_params['f0']) == len(synth_params['amps'])


def generate_and_save_synth_params(synthesis_generator, dataset, output_dir):
  for data in tqdm(dataset):
    if data['f0_hz'].shape[1] != data['loudness_db'].shape[1]:
      data['loudness_db'] = make_same_length(data['loudness_db'], data['f0_hz'])

    synth_params = synthesis_generator.synth_coder(data, training=False)

    # transfer to normalized synth_params

    processor_group = get_process_group(data['f0_hz'].shape[1])

    synth_params = processor_group.get_controls(synth_params, verbose=False)
    synth_audio = processor_group.get_signal(synth_params)
    synth_audio = synthesis_generator.reverb_module(synth_audio,
                                                    reverb_number=data[
                                                      'instrument_id'])
    data['synth_audio'] = synth_audio

    f0, amps, hd, noise = extract_harm_controls(synth_params)
    synth_params = {'f0': f0, 'amps': amps, 'hd': hd, 'noise': noise}

    save_synth_params(data, synth_params, output_dir)


# Create dataset contains synthesis parameters.

def get_piece_note_features(data, conditioning):
  note_feature_piece = []
  onset_index = np.where(data['onsets'] == 1)[0]
  offset_index = np.where(data['offsets'] == 1)[0]
  assert len(onset_index) == len(offset_index)
  total_frame = conditioning['amplitude'].shape[1]
  last_off = -1

  for on, off in zip(onset_index, offset_index):
    if on != off:
      amp_mean = np.mean(conditioning['amplitude'][0, on:off + 1, :])
      amp_std = np.std(conditioning['amplitude'][0, on:off + 1, :])
      brightness = np.mean(conditioning['brightness'][0, on:off + 1, :])
      attack_level = np.mean(
        conditioning['noise_level'][0, on:min(off + 1, on + 10), :])

      pv = conditioning['pitch_deviation'][0, on:off + 1, :]
      note_mask = tf.ones([1, pv.shape[0], 1])
      note_mask_pad = to_length(note_mask, dst_length=1000)
      pv_pad = to_length(pv[tf.newaxis, ...], dst_length=1000)
      frame_wise_vibrato_rate, frame_wise_vibrato_extend = get_vibrato_feature(
        pv_pad, note_mask_pad)
      vibrato_rate = frame_wise_vibrato_rate[0, 0, 0].numpy()
      vibrato_extend = frame_wise_vibrato_extend[0, 0, 0].numpy()

      amps = conditioning['amplitude'][0, on:off + 1, :]
      note_mask = tf.ones([1, pv.shape[0], 1])
      note_mask_pad = to_length(note_mask, dst_length=1000)
      amps_pad = to_length(amps[tf.newaxis, ...], dst_length=1000)
      amplitudes_max_pos = get_amplitudes_max_pos(amps_pad, note_mask_pad)
      amplitudes_max_pos = amplitudes_max_pos[0, 0, 0].numpy()

      # scale conditioning so that most value are in [0, 1]
      # TODO: (yusongwu) enable automatic scaling and merge with
      #  get_conditioning_dict
      amp_mean = np.where(np.equal(amp_mean, 0.0), 0.0, amp_mean / 60 + 1.5)
      amp_std *= (2.5 / 60)
      vibrato_extend *= 10
      brightness *= 5
      attack_level = np.where(np.equal(attack_level, 0.0), 0.0,
                              attack_level / 40 + 2.625)

      note_feature_dict = {
        'amplitude_mean': amp_mean,
        'amplitude_std': amp_std,
        'vibrato_extend': vibrato_extend,
        'brightness': brightness,
        'attack_level': attack_level,
        'amplitudes_max_pos': amplitudes_max_pos,
        'note_length': off - on + 1,
        'note_pitch': data['midi'][on],
        'instrument_id': data['instrument_id'].item()
      }

      assert note_feature_dict['note_pitch'] != 0

      if on - last_off == 0:
        note_feature_dict['note_length'] -= 1

      elif on - last_off > 1:  # add rest note
        note_feature_dict_rest = {k: 0 for k in note_feature_dict.keys()}
        note_feature_dict_rest['note_length'] = on - last_off - 1
        note_feature_dict_rest['note_pitch'] = 0
        note_feature_dict_rest['instrument_id'] = note_feature_dict[
          'instrument_id']
        note_feature_piece.append(note_feature_dict_rest)

      last_off = off

      if note_feature_dict['note_length'] != 0:
        note_feature_piece.append(note_feature_dict)

  if last_off < (total_frame - 1):  # add the last rest note
    note_feature_dict_rest = {k: 0 for k in note_feature_dict.keys()}
    note_feature_dict_rest['note_length'] = total_frame - last_off - 1
    note_feature_dict_rest['note_pitch'] = 0
    note_feature_dict_rest['instrument_id'] = note_feature_dict['instrument_id']
    note_feature_piece.append(note_feature_dict_rest)

  total_frame_note = sum([note['note_length'] for note in note_feature_piece])
  assert total_frame_note == total_frame

  return note_feature_piece


def note_features_to_input_dict(note_features):
  note_features = copy.deepcopy(note_features)
  note_pitch = note_features['note_pitch']
  note_length = note_features['note_length']
  instrument_id = note_features['instrument_id']
  note_features.pop('note_pitch')
  note_features.pop('note_length')
  note_features.pop('instrument_id')
  conditioning_feature = np.array([v for v in note_features.values()])
  input_dict = {
    'note_pitch': note_pitch,
    'note_length': note_length,
    'conditioning_feature': conditioning_feature,
    'instrument_id': instrument_id,
  }
  return input_dict


def segment_single(data, window_length, hop_length):
  if len(data) < window_length:
    return [data]
  data_list = [data[i:i + window_length] for i in
               range(0, len(data) - window_length + hop_length, hop_length)]
  if len(data_list[-1]) < window_length:
    # set the last one as same length disregarding hop_length
    data_list[-1] = data[len(data) - window_length:]
  return data_list


def segment_data(note_features_all, window_length=64, hop_length=1):
  data_segmented_all = []
  for note_feature_piece in note_features_all:
    input_dict_piece = [note_features_to_input_dict(n) for n in
                        note_feature_piece]
    data_segmented = segment_single(input_dict_piece,
                                    window_length=window_length,
                                    hop_length=hop_length)
    data_segmented_all.extend(data_segmented)
  return data_segmented_all


def get_all_note_features(data_dir):
  midi_file_list = glob.glob(data_dir + '/*.midi.npy')
  note_features_all = []
  audio_all = []
  for midi_file in tqdm(midi_file_list):
    data = np.load(midi_file.replace('.midi', '.synth_params'),
                   allow_pickle=True).item()
    data.update({'midi': np.load(midi_file)})
    data.update({'onsets': np.load(midi_file.replace('.midi', '.onsets'))})
    data.update({'offsets': np.load(midi_file.replace('.midi', '.offsets'))})
    data.update(
      {'instrument_id': np.load(midi_file.replace('.midi', '.instrument_id'))})
    conditioning = get_interpretable_conditioning(
      ddsp.core.midi_to_hz(data['midi'][np.newaxis, :, np.newaxis],
                           midi_zero_silence=True),
      data['f0'][np.newaxis, ...],
      data['amps'][np.newaxis, ...],
      data['hd'][np.newaxis, ...],
      data['noise'][np.newaxis, ...])
    note_feature_piece = get_piece_note_features(data, conditioning)
    note_features_all.append(note_feature_piece)
    audio_all.append(np.load(midi_file.replace('.midi', '.wav')))
  return note_features_all, audio_all


def get_stat_str(value_list):
  return f'Max: {np.max(value_list):.2f}, ' \
         f'Min: {np.min(value_list):.2f}, ' \
         f'Mean: {np.mean(value_list):.2f}, ' \
         f'Std: {np.std(value_list):.2f}'


def plot_save_expression_stats(key, note_params_reduced, show_plot=True,
                               save_fig=False, output_dir=None):
  value_list = [n[key] for n in note_params_reduced]
  print(f'{key} {get_stat_str(value_list)}')
  plt.hist(value_list, bins=50)
  if show_plot:
    plt.show()
  if save_fig:
    plt.savefig(os.path.join(output_dir, f'{key}_stat.png'))


def plot_stats(note_params_reduced, instrument_id=None, show_plot=True,
               save_fig=False, output_dir=None):
  if save_fig and not output_dir:
    raise ValueError(
      'Please specify output directory for saving plot statistics.')
  if instrument_id is None:
    note_params_reduced = [n for n in note_params_reduced if
                           n['note_pitch'] != 0]
  else:
    note_params_reduced = [n for n in note_params_reduced if
                           (n['note_pitch'] != 0 and n[
                             'instrument_id'] == instrument_id)]

  plot_save_expression_stats('vibrato_extend', note_params_reduced,
                             show_plot=show_plot,
                             save_fig=save_fig, output_dir=output_dir)

  plot_save_expression_stats('amplitude_mean', note_params_reduced,
                             show_plot=show_plot,
                             save_fig=save_fig, output_dir=output_dir)

  plot_save_expression_stats('amplitude_std', note_params_reduced,
                             show_plot=show_plot,
                             save_fig=save_fig, output_dir=output_dir)

  plot_save_expression_stats('brightness', note_params_reduced,
                             show_plot=show_plot,
                             save_fig=save_fig, output_dir=output_dir)

  plot_save_expression_stats('attack_level', note_params_reduced,
                             show_plot=show_plot,
                             save_fig=save_fig, output_dir=output_dir)

  plot_save_expression_stats('note_pitch', note_params_reduced,
                             show_plot=show_plot,
                             save_fig=save_fig, output_dir=output_dir)

  plot_save_expression_stats('note_length', note_params_reduced,
                             show_plot=show_plot,
                             save_fig=save_fig, output_dir=output_dir)


def preprocess_tensors(data):
  data_out = {}
  data_out['note_pitch'] = tf.convert_to_tensor(
    np.array([n['note_pitch'] for n in data]))
  data_out['note_length'] = tf.convert_to_tensor(
    np.array([n['note_length'] for n in data]))
  data_out['conditioning_feature'] = tf.convert_to_tensor(
    np.array([n['conditioning_feature'] for n in data]))
  data_out['instrument_id'] = tf.convert_to_tensor(
    np.array(data[0]['instrument_id']))
  return data_out


def make_dataset_no_segment(note_features_all):
  dataset_not_segmented = []
  for note_feature_piece in note_features_all:
    input_dict_piece = [note_features_to_input_dict(n) for n in
                        note_feature_piece]
    dataset_not_segmented.append(preprocess_tensors(input_dict_piece))
  return dataset_not_segmented


def make_and_save_pickles(data_dir, split, pickle_output_dir,
                          stats_plot_output_dir):
  split_stats_plot_output_dir = os.path.join(stats_plot_output_dir, f'{split}')
  os.makedirs(split_stats_plot_output_dir)
  note_features_all, audio_all = get_all_note_features(data_dir)
  note_features_all_reduced = []
  for n in note_features_all:
    note_features_all_reduced.extend(n)
  plot_stats(note_features_all_reduced, show_plot=False, save_fig=True,
             output_dir=split_stats_plot_output_dir)
  data_language_model = segment_data(note_features_all)
  data_language_model_preprocessed = [preprocess_tensors(d) for d in
                                      data_language_model]
  data_language_model_separate_piece = [segment_data([n]) for n in
                                        note_features_all]
  data_language_model_separate_piece_preprocessed = []
  for data in data_language_model_separate_piece:
    data_language_model_separate_piece_preprocessed.append(
      [preprocess_tensors(d) for d in data])
  not_segmented = make_dataset_no_segment(note_features_all)
  print('Total number of training pieces:', len(note_features_all))
  print('Total number of training samples:',
        len(data_language_model_preprocessed))

  pickle_dump(data_language_model_preprocessed,
              os.path.join(pickle_output_dir, f'{split}.pickle'))
  pickle_dump(data_language_model_separate_piece_preprocessed,
              os.path.join(pickle_output_dir, f'{split}_separate_piece.pickle'))
  pickle_dump(not_segmented,
              os.path.join(pickle_output_dir, f'{split}_not_segmented.pickle'))
  pickle_dump(audio_all,
              os.path.join(pickle_output_dir, f'{split}_audio_all.pickle'))


def dump_expression_generator_dataset(model, data_dir, output_dir):
  # TODO: (yusongwu) add automatic note expression scaling

  synth_params_output_dir = os.path.join(output_dir, 'synth_params')
  pickle_output_dir = os.path.join(output_dir, 'pickles')
  stats_plot_output_dir = os.path.join(output_dir, 'stats_plot')
  os.makedirs(pickle_output_dir, exist_ok=True)
  os.makedirs(stats_plot_output_dir, exist_ok=True)

  test_data_loader = UrmpMidiUnbatched(data_dir, instrument_key='all',
                                       split='test', suffix='unbatched')
  test_dataset = test_data_loader.get_batch(batch_size=1, shuffle=True,
                                            repeats=1)
  train_data_loader = UrmpMidiUnbatched(data_dir, instrument_key='all',
                                        split='train', suffix='unbatched')
  train_dataset = train_data_loader.get_batch(batch_size=1, shuffle=True,
                                              repeats=1)

  train_set_output_dir = os.path.join(synth_params_output_dir, 'train')
  os.makedirs(train_set_output_dir, exist_ok=True)
  generate_and_save_synth_params(model, train_dataset, train_set_output_dir)
  test_set_output_dir = os.path.join(synth_params_output_dir, 'test')
  os.makedirs(test_set_output_dir, exist_ok=True)
  generate_and_save_synth_params(model, test_dataset, test_set_output_dir)

  make_and_save_pickles(train_set_output_dir, 'train', pickle_output_dir,
                        stats_plot_output_dir)
  make_and_save_pickles(test_set_output_dir, 'test', pickle_output_dir,
                        stats_plot_output_dir)
