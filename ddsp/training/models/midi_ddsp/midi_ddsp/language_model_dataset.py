
import tensorflow as tf
import numpy as np
from itertools import cycle

_AUTOTUNE = tf.data.experimental.AUTOTUNE


def random_choose_data(data):
  def gen():
    for piece in cycle(data):
      j = np.random.randint(0, len(piece))
      yield piece[j]

  return gen


def get_lang_model_dataset(data_language_model_preprocessed, batch_size=256,
                           repeats=-1):
  # np.random.shuffle(data_language_model_preprocessed)
  output_types = {
    'note_pitch': tf.int64,
    'note_length': tf.int64,
    'conditioning_feature': tf.float32,
    'instrument_id': tf.int64,
  }
  padded_shapes = {
    'note_pitch': [None],
    'note_length': [None],
    'conditioning_feature': [None, None],
    'instrument_id': [],
  }

  def _reshape_tensors(data):
    cond = {
      'note_pitch': data['note_pitch'],
      'note_length': tf.cast(data['note_length'], tf.float32)[
                       ..., tf.newaxis] * 0.004,
      'instrument_id': data['instrument_id']
    }
    target = data['conditioning_feature']
    scale = np.ones((1, 6))  # HACK
    scale = tf.convert_to_tensor(scale, tf.float32)
    target *= scale

    return {
      'cond': cond,
      'target': target,
    }

  if repeats != 1:
    data_iter = random_choose_data(data_language_model_preprocessed)
    dataset = tf.data.Dataset.from_generator(data_iter,
                                             output_types=output_types)

  else:
    dataset = tf.data.Dataset.from_generator(
      lambda: data_language_model_preprocessed, output_types=output_types)

  dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
  dataset = dataset.repeat(repeats)
  dataset = dataset.map(_reshape_tensors, num_parallel_calls=_AUTOTUNE)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

  return dataset
