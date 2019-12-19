# Copyright 2019 The DDSP Authors.
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

# Lint as: python3
"""Helper functions for running DDSP colab notebooks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import io

import ddsp
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import tensorflow.compat.v1 as tf

DEFAULT_SAMPLE_RATE = 16000
_play_id = 0  # Used for ephemeral colab_play.


def play(array_of_floats,
         sample_rate=DEFAULT_SAMPLE_RATE,
         ephemeral=True,
         autoplay=False):
  """Creates an HTML5 audio widget to play a sound in Colab.

  This function should only be called from a Colab notebook.

  Args:
    array_of_floats: A 1D or 2D array-like container of float sound samples.
      Values outside of the range [-1, 1] will be clipped.
    sample_rate: Sample rate in samples per second.
    ephemeral: If set to True, the widget will be ephemeral, and disappear on
      reload (and it won't be counted against realtime document size).
    autoplay: If True, automatically start playing the sound when the widget is
      rendered.
  """
  from google.colab.output import _js_builder as js  # pylint:disable=g-import-not-at-top,protected-accessk,import-error

  normalizer = float(np.iinfo(np.int16).max)
  array_of_ints = np.array(
      np.asarray(array_of_floats) * normalizer, dtype=np.int16)
  memfile = io.BytesIO()
  wavfile.write(memfile, sample_rate, array_of_ints)
  html = """<audio controls {autoplay}>
              <source controls src="data:audio/wav;base64,{base64_wavfile}"
              type="audio/wav" />
              Your browser does not support the audio element.
            </audio>"""
  html = html.format(
      autoplay='autoplay' if autoplay else '',
      base64_wavfile=base64.b64encode(memfile.getvalue()).decode('ascii'))
  memfile.close()
  global _play_id
  _play_id += 1
  if ephemeral:
    element = 'id_%s' % _play_id
    display.display(display.HTML('<div id="%s"> </div>' % element))
    js.Js('document', mode=js.EVAL).getElementById(element).innerHTML = html
  else:
    display.display(display.HTML(html))


def specplot(audio, vmin=-5, vmax=1, rotate=True, size=512 + 256):
  """Plot the log magnitude spectrogram of audio."""
  with tf.Session() as sess:
    logmag = sess.run(
        ddsp.spectral_ops.calc_logmag(ddsp.core.f32(audio), size=size))
  if rotate:
    logmag = np.rot90(logmag)
  # Plotting.
  plt.matshow(logmag, vmin=vmin, vmax=vmax, cmap=plt.cm.magma, aspect='auto')
  plt.xticks([])
  plt.yticks([])
  plt.xlabel('Time')
  plt.ylabel('Frequency')


def transfer_function(ir, sample_rate=DEFAULT_SAMPLE_RATE):
  """Get true transfer function from an impulse_response."""
  n_fft = ddsp.core.get_fft_size(0, ir.shape.as_list()[-1])
  frequencies = np.abs(np.fft.fftfreq(n_fft, 1/sample_rate)[:int(n_fft/2) + 1])
  magnitudes = tf.abs(tf.signal.rfft(ir, [n_fft]))
  return frequencies, magnitudes


def plot_impulse_responses(impulse_response,
                           desired_magnitudes,
                           sample_rate=DEFAULT_SAMPLE_RATE):
  """Plot a target frequency response, and that of an impulse response."""
  n_fft = desired_magnitudes.shape[-1] * 2
  frequencies = np.fft.fftfreq(n_fft, 1/sample_rate)[:n_fft//2]

  with tf.Session() as sess:
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    # Desired transfer function.
    plt.semilogy(frequencies, desired_magnitudes, label='Desired')
    # True transfer function.
    true_frequencies, true_magnitudes = transfer_function(impulse_response)
    plt.semilogy(true_frequencies,
                 sess.run(true_magnitudes)[0, 0, :], label='True')
    plt.title('Transfer Function')
    plt.legend()

    plt.subplot(122)
    plt.plot(sess.run(impulse_response)[0, 0, :])
    plt.title('Impulse Response')
