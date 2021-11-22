"""Urmp dataset using unsegmented data."""

from ddsp.training.data import Urmp
import tensorflow as tf

_AUTOTUNE = tf.data.experimental.AUTOTUNE


class UrmpMidiUnbatched(Urmp):
  """Urmp dataset using unsegmented data. "Unsegmented" here means that the data
  samples are not segmented to 4-second chunks as in UrmpMidi dataset.
  """
  # TODO: (yusongwu) merge this into ddsp.training.data

  _INSTRUMENTS = ['vn', 'va', 'vc', 'db', 'fl', 'ob', 'cl', 'sax', 'bn', 'tpt',
                  'hn', 'tbn', 'tba']

  @property
  def features_dict(self):
    base_features = {
      'audio':
        tf.io.VarLenFeature(dtype=tf.float32),
      'f0_hz':
        tf.io.VarLenFeature(dtype=tf.float32),
      'f0_confidence':
        tf.io.VarLenFeature(dtype=tf.float32),
      'loudness_db':
        tf.io.VarLenFeature(dtype=tf.float32),
    }
    base_features.update({
      'note_active_velocities':
        tf.io.VarLenFeature(tf.float32),
      'note_active_frame_indices':
        tf.io.VarLenFeature(tf.float32),
      'instrument_id': tf.io.FixedLenFeature([], tf.string),
      'recording_id': tf.io.FixedLenFeature([], tf.string),
      'power_db':
        tf.io.VarLenFeature(dtype=tf.float32),
      'note_onsets':
        tf.io.VarLenFeature(dtype=tf.float32),
      'note_offsets':
        tf.io.VarLenFeature(dtype=tf.float32),
    })
    return base_features

  def get_dataset(self, shuffle=True):
    instrument_ids = range(len(self._INSTRUMENTS))
    inst_vocab = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(self._INSTRUMENTS, instrument_ids),
      -1)

    def _reshape_tensors(data):
      data['audio'] = tf.sparse.to_dense(data['audio'])
      data['note_active_frame_indices'] = tf.reshape(
        tf.sparse.to_dense(data['note_active_frame_indices']), (-1, 128))
      data['note_active_velocities'] = tf.reshape(
        tf.sparse.to_dense(data['note_active_velocities']), (-1, 128))
      data['instrument_id'] = inst_vocab.lookup(data['instrument_id'])
      data['midi'] = tf.argmax(data['note_active_frame_indices'], axis=-1)
      data['f0_hz'] = tf.sparse.to_dense(data['f0_hz'])[..., tf.newaxis]
      data['loudness_db'] = tf.sparse.to_dense(data['loudness_db'])[
        ..., tf.newaxis]
      # reshape and rename things
      onsets = tf.reduce_sum(
        tf.reshape(tf.sparse.to_dense(data['note_onsets']), (-1, 128)), axis=-1)
      data['onsets'] = tf.cast(onsets > 0, tf.int64)
      offsets = tf.reduce_sum(
        tf.reshape(tf.sparse.to_dense(data['note_offsets']), (-1, 128)),
        axis=-1)
      data['offsets'] = tf.cast(offsets > 0, tf.int64)

      return data

    ds = super().get_dataset(shuffle)
    ds = ds.map(_reshape_tensors, num_parallel_calls=_AUTOTUNE)
    return ds
