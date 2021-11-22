
from abc import ABC
import tensorflow as tf
import tensorflow.keras.layers as layers
import ddsp.training.nn as nn
from data_handling.instrument_name_utils import NUM_INST
from midi_ddsp.cond_rnn import TwoLayerCondAutoregRNN


class LangModelOutputLayer(layers.Layer):
  def __init__(self, n_out, nhid=256):
    super(LangModelOutputLayer, self).__init__()
    self.n_out = n_out
    self.dense_out = nn.FcStackOut(ch=nhid, layers=2, n_out=self.n_out)

  def call(self, inputs):
    outputs = self.dense_out(inputs)
    outputs = {'raw_output': outputs}
    return outputs


class InterpCondAutoregressiveRNN(TwoLayerCondAutoregRNN, tf.keras.Model, ABC):
  # TODO：(yusongwu) merge teacher_force and autoregressive function,
  # things need to change:
  # def sample_out(self, out, cond, time, training=False):
  # curr_out = self.sample_out(curr_out, cond, i, training=training)
  # output = self.split_teacher_force_output(output, cond)
  def __init__(self, n_out=6, nhid=128, norm=True, dropout=0.5):
    super(InterpCondAutoregressiveRNN, self).__init__(
      nhid=nhid,
      n_out=n_out,
      input_dropout=True,
      input_dropout_p=0.5,
      dropout=dropout,
    )
    self.birnn = layers.Bidirectional(tf.keras.layers.GRU(
      units=nhid, return_sequences=True, dropout=dropout
    ), )
    self.dense_out = LangModelOutputLayer(nhid=nhid, n_out=self.n_out)
    self.norm = nn.Normalize('layer') if norm else None
    self.pitch_emb = layers.Embedding(128, 64)
    self.duration_emb = layers.Dense(64)
    self.instrument_emb = layers.Embedding(NUM_INST, 64)

  def autoregressive(self, cond, training=False):
    note_pitch = cond['note_pitch'][..., tf.newaxis]
    cond = self.encode_cond(cond, training=training)
    batch_size = cond.shape[0]
    length = cond.shape[1]
    prev_out = tf.tile([[0.0]], [batch_size, self.n_out])[:, tf.newaxis,
               :]  # go_frame
    prev_states = (None, None)
    overall_outputs = []

    for i in range(length):
      curr_cond = cond[:, i, :][:, tf.newaxis, :]
      prev_out = self.encode_out(prev_out)
      curr_out, curr_states = self._one_step(curr_cond, prev_out, prev_states,
                                             training=training)
      curr_out = self.sample_out(curr_out,
                                 note_pitch[:, i, :][:, tf.newaxis, :])
      overall_outputs.append(curr_out)
      prev_out, prev_states = curr_out['output'], curr_states

    outputs = {}
    for k in curr_out.keys():
      outputs[k] = tf.concat([x[k] for x in overall_outputs], axis=1)
    return outputs

  def teacher_force(self, cond, out, training=True):
    note_pitch = cond['note_pitch'][..., tf.newaxis]
    out_shifted = self.right_shift_encode_out(out)
    cond = self.encode_cond(cond, training=training)
    z_in = tf.concat([cond, out_shifted], axis=-1)
    z_out, *states = self.rnn1(z_in, training=training)
    z_out, *states = self.rnn2(z_out, training=training)
    output = self.decode_out(z_out)
    output = self.sample_out(output, note_pitch)
    return output

  def encode_cond(self, cond, training=False):
    z_pitch = self.pitch_emb(cond['note_pitch'])
    z_duration = self.duration_emb(cond['note_length'])
    z_instrument = self.instrument_emb(
      tf.tile(cond['instrument_id'][:, tf.newaxis], [1, z_pitch.shape[1]]))
    cond = tf.concat([z_pitch, z_duration, z_instrument], axis=-1)
    cond = self.birnn(cond, training=training)
    return cond

  def decode_out(self, z_out):
    if self.norm is not None:
      z_out = self.norm(z_out)
    output = self.dense_out(z_out)
    return output

  def sample_out(self, out, note_pitch):
    output = out.copy()
    sampled_output = out['raw_output']
    rest_note_mask = tf.cast(note_pitch != 0, tf.float32)
    sampled_output *= rest_note_mask
    output['output'] = sampled_output
    return output

  def call(self, cond, out=None, training=False):

    if training:
      outputs = self.teacher_force(cond, out, training=training)
    else:
      outputs = self.autoregressive(cond, training=training)
    return outputs
