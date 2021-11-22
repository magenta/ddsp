"""Abstract model for conditional autoregressive RNN used in MIDI-DDSP."""

import tensorflow as tf
from tqdm import tqdm

tfk = tf.keras
tfkl = tfk.layers


class StackedRNN(tfkl.Layer):
  """Stacked RNN implementated using tfkl.Layers."""

  def __init__(self, nhid=256, nlayers=2, rnn_type='gru', dropout=0.5):
    super(StackedRNN, self).__init__()
    if rnn_type == 'gru':
      rnn_layer = tfkl.GRU
    elif rnn_type == 'lstm':
      rnn_layer = tfkl.LSTM
    else:
      raise TypeError('Unknown rnn_type')
    self.nhid = nhid
    self.nlayers = nlayers
    self.net = {
      str(i): rnn_layer(nhid, return_sequences=True, return_state=True,
                        dropout=dropout) for i in
      range(nlayers)}

  def call(self, x, initial_state=None, training=False):
    states_out_all = []
    z_out = x
    if initial_state is None:
      initial_state = [None for _ in range(self.nlayers)]
    for i in range(self.nlayers):
      z_out, states_out = self.net[str(i)](z_out,
                                           initial_state=initial_state[i],
                                           training=training)
      states_out_all.append(states_out)
    return z_out, states_out_all


class TwoLayerCondAutoregRNN(tfkl.Layer):
  """Conditional two-layer autoregressive RNN.
  The RNN here takes input from a conditioning sequence and its previous output.
  The RNN is trained using teacher forcing and inference in autoregressive mode.
  """

  def __init__(self, nhid, n_out, input_dropout=True, input_dropout_p=0.5,
               dropout=0.5, rnn_type='gru'):
    """Constructor."""
    super(TwoLayerCondAutoregRNN, self).__init__()
    self.n_out = n_out
    self.nhid = nhid
    self.input_dropout = input_dropout
    self.input_dropout_p = input_dropout_p
    if rnn_type == 'gru':
      self.rnn1 = tfkl.GRU(nhid, return_sequences=True, return_state=True,
                           dropout=dropout)
      self.rnn2 = tfkl.GRU(nhid, return_sequences=True, return_state=True,
                           dropout=dropout)
    elif rnn_type == 'lstm':
      self.rnn1 = tfkl.LSTM(nhid, return_sequences=True, return_state=True,
                            dropout=dropout)
      self.rnn2 = tfkl.LSTM(nhid, return_sequences=True, return_state=True,
                            dropout=dropout)
    else:
      raise ValueError('Unknown RNN type.')

  def _one_step(self, curr_cond, prev_out, prev_states, training=False):
    """One step inference."""
    prev_states_1, prev_states_2 = prev_states
    curr_z_in = tf.concat([curr_cond, prev_out],
                          axis=-1)  # [batch_size, 1, dim]
    curr_z_in = self.encode_z(curr_z_in, training=training)
    curr_z_out, curr_states_1 = self.rnn1(curr_z_in,
                                          initial_state=prev_states_1,
                                          training=training)
    curr_z_out, curr_states_2 = self.rnn2(curr_z_out,
                                          initial_state=prev_states_2,
                                          training=training)
    curr_out = self.decode_out(curr_z_out)
    return curr_out, (curr_states_1, curr_states_2)

  def autoregressive(self, cond, training=False, display_progressbar=False):
    """Autoregressive inference."""
    cond_encode = self.encode_cond(cond)
    batch_size = cond_encode.shape[0]
    length = cond_encode.shape[1]
    prev_out = tf.tile([[0.0]], [batch_size, self.n_out])[:, tf.newaxis,
               :]  # go_frame
    prev_states = (None, None)
    overall_outputs = []

    for i in tqdm(range(length), position=0, leave=True, desc="Generating: ",
                  disable=not display_progressbar):
      curr_cond = cond_encode[:, i, :][:, tf.newaxis, :]
      prev_out = self.encode_out(prev_out, training=training)
      curr_out, curr_states = self._one_step(curr_cond, prev_out, prev_states,
                                             training=training)
      curr_out = self.sample_out(curr_out, training=training)
      overall_outputs.append(curr_out)
      prev_out, prev_states = curr_out['curr_out'], curr_states

    outputs = {}
    for k in curr_out.keys():
      outputs[k] = tf.concat([x[k] for x in overall_outputs], axis=1)
    return outputs

  def right_shift_encode_out(self, out, training=False):
    """Right shift the ground truth target by one timestep and encode."""
    go_frame = tf.tile([[0.0]], [out.shape[0], out.shape[-1]])[:, tf.newaxis, :]
    out = tf.concat([go_frame, out[:, :-1, :]], axis=1)
    out = self.encode_out(out, training=training)
    # input dropout
    if self.input_dropout:
      input_dropout_mask = tf.cast(
        tf.random.uniform(
          shape=[out.shape[0], out.shape[1], 1]) > self.input_dropout_p,
        tf.float32)
      out = out * input_dropout_mask
    return out

  def teacher_force(self, cond, out, training=True):
    """Run teacher force."""
    out_shifted = self.right_shift_encode_out(out, training=training)
    cond = self.encode_cond(cond, training=training)
    z_in = tf.concat([cond, out_shifted], axis=-1)
    z_in = self.encode_z(z_in, training=training)
    z_out, *states = self.rnn1(z_in, training=training)
    z_out, *states = self.rnn2(z_out, training=training)
    output = self.decode_out(z_out, training=training)
    output = self.split_teacher_force_output(output)
    return output

  def encode_z(self, z, training=False):
    return z

  def encode_out(self, out, training=False):
    return out

  def encode_cond(self, cond, training=False):
    return cond

  def decode_out(self, z_out, training=False):
    return z_out

  def sample_out(self, out, training=False):
    return out

  def preprocess(self, cond, out):
    return cond, out

  def postprocess(self, outputs, cond, training=False):
    return outputs

  def split_teacher_force_output(self, output):
    return output

  def call(self, cond, out=None, training=False):
    """Forward call."""
    if training:
      cond, out = self.preprocess(cond, out)
      outputs = self.teacher_force(cond, out, training=training)
    else:
      cond, out = self.preprocess(cond, out)
      outputs = self.autoregressive(cond, training=training)

    outputs = self.postprocess(outputs, cond, training=training)

    return outputs
