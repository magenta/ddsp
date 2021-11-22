"""Utility functions used in training loop."""

import numpy as np
import inspect
import tensorflow as tf
import random
import os
import argparse
import librosa
import matplotlib.pyplot as plt

from utils.audio_io import save_wav


def plot_spec(wav, sr, title='', vmin=-8, vmax=1, save_path=None):
  """Plot and save the spectrogram."""
  D = np.log(np.abs(librosa.stft(wav, n_fft=512 + 256)))
  librosa.display.specshow(D, sr=sr, vmin=vmin, vmax=vmax, cmap='magma')
  plt.title(title)
  if save_path:
    plt.savefig(save_path)
    plt.close()


def get_hp(file):
  """Retrieve hyperparameters from log files."""
  log_all = open(file, "r").readlines()
  for line in log_all:
    if '{\'add_synth_loss\':' in line:
      hp = eval(line.strip()[32:])
      break
  return hp


def set_seed(seed):
  """Set the random seed."""
  os.environ['PYTHONHASHSEED'] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)
  os.environ['TF_DETERMINISTIC_OPS'] = '1'


def print_hparams(hp):
  """Print the hyperparameters"""
  attributes = inspect.getmembers(hp, lambda a: not (inspect.isroutine(a)))
  return dict([a for a in attributes if
               not (a[0].startswith('__') and a[0].endswith('__'))])


def save_results(pred_batch, ref_wav_batch, log_dir, save_folder_name,
                 sample_rate, acoustic_params=None,
                 separate_signal=None, file_name=None, bottleneck_feature=None,
                 note=None, no_pred_suffix=False):
  """Save the evaluation results."""
  if file_name and pred_batch.shape[0] > 1:
    raise RuntimeError('Got a file name to save, but has batch_size > 1')
  pred = pred_batch.numpy()
  if ref_wav_batch is not None:
    ref_wav = ref_wav_batch.numpy()
  if log_dir is None:
    save_dir = save_folder_name
  else:
    save_dir = os.path.join(log_dir,
                            f'results_{os.path.basename(save_folder_name)}')
  os.makedirs(save_dir, exist_ok=True)
  output_features_save_dir = os.path.join(save_dir, 'output_features')
  separate_signal_save_dir = os.path.join(save_dir, 'separate_signal')
  spectrogram_dir = os.path.join(save_dir, 'spectrogram')
  os.makedirs(output_features_save_dir, exist_ok=True)
  os.makedirs(separate_signal_save_dir, exist_ok=True)
  os.makedirs(spectrogram_dir, exist_ok=True)
  for i in range(pred.shape[0]):

    save_file_name = file_name if file_name else i

    pred_file_suffix = '' if no_pred_suffix else '_pred'
    save_wav(pred[i],
             os.path.join(save_dir, f'{save_file_name}{pred_file_suffix}.wav'),
             sample_rate=sample_rate)
    plot_spec(pred[i], sr=sample_rate,
              title=f'{save_file_name}{pred_file_suffix}.wav',
              save_path=os.path.join(spectrogram_dir,
                                     f'{save_file_name}{pred_file_suffix}.png'))
    if ref_wav_batch is not None:
      save_wav(ref_wav[i], os.path.join(save_dir, f'{save_file_name}_ref.wav'),
               sample_rate=sample_rate)
      plot_spec(ref_wav[i], sr=sample_rate, title=f'{save_file_name}_ref.wav',
                save_path=os.path.join(spectrogram_dir,
                                       f'{save_file_name}_ref.png'))
    if acoustic_params:
      for k in acoustic_params.keys():
        np.save(
          os.path.join(output_features_save_dir, f'{save_file_name}_{k}.npy'),
          acoustic_params[k][i].numpy())
    if separate_signal:
      for k in separate_signal.keys():
        save_wav(separate_signal[k][i].numpy(),
                 os.path.join(separate_signal_save_dir,
                              f'{save_file_name}_{k}.wav'),
                 sample_rate=sample_rate)
    if bottleneck_feature:
      for k in bottleneck_feature.keys():
        if 'loss' not in k and 'perplexity' not in k:
          np.save(
            os.path.join(output_features_save_dir, f'{save_file_name}_{k}.npy'),
            bottleneck_feature[k][i].numpy())
    if note:
      for k in note.keys():
        np.save(
          os.path.join(output_features_save_dir, f'{save_file_name}_{k}.npy'),
          note[k][i].numpy())


def str2bool(v):
  """Enable boolean in argparse by passing string."""
  # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')
