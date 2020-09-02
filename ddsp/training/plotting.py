# Copyright 2020 The DDSP Authors.
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
"""Plotting utilities for the DDSP library. Useful in colab and elsewhere."""

from ddsp import core
from ddsp import spectral_ops
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v2 as tf

DEFAULT_SAMPLE_RATE = spectral_ops.CREPE_SAMPLE_RATE


def specplot(audio,
             vmin=-5,
             vmax=1,
             rotate=True,
             size=512 + 256,
             **matshow_kwargs):
  """Plot the log magnitude spectrogram of audio."""
  # If batched, take first element.
  if len(audio.shape) == 2:
    audio = audio[0]

  logmag = spectral_ops.compute_logmag(core.tf_float32(audio), size=size)
  if rotate:
    logmag = np.rot90(logmag)
  # Plotting.
  plt.matshow(logmag,
              vmin=vmin,
              vmax=vmax,
              cmap=plt.cm.magma,
              aspect='auto',
              **matshow_kwargs)
  plt.xticks([])
  plt.yticks([])
  plt.xlabel('Time')
  plt.ylabel('Frequency')


def transfer_function(ir, sample_rate=DEFAULT_SAMPLE_RATE):
  """Get true transfer function from an impulse_response."""
  n_fft = core.get_fft_size(0, ir.shape.as_list()[-1])
  frequencies = np.abs(
      np.fft.fftfreq(n_fft, 1 / sample_rate)[:int(n_fft / 2) + 1])
  magnitudes = tf.abs(tf.signal.rfft(ir, [n_fft]))
  return frequencies, magnitudes


def plot_impulse_responses(impulse_response,
                           desired_magnitudes,
                           sample_rate=DEFAULT_SAMPLE_RATE):
  """Plot a target frequency response, and that of an impulse response."""
  n_fft = desired_magnitudes.shape[-1] * 2
  frequencies = np.fft.fftfreq(n_fft, 1 / sample_rate)[:n_fft // 2]
  true_frequencies, true_magnitudes = transfer_function(impulse_response)

  # Plot it.
  plt.figure(figsize=(12, 6))
  plt.subplot(121)
  # Desired transfer function.
  plt.semilogy(frequencies, desired_magnitudes, label='Desired')
  # True transfer function.
  plt.semilogy(true_frequencies, true_magnitudes[0, 0, :], label='True')
  plt.title('Transfer Function')
  plt.legend()

  plt.subplot(122)
  plt.plot(impulse_response[0, 0, :])
  plt.title('Impulse Response')


