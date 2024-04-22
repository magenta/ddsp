# Copyright 2024 The DDSP Authors.
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

"""Plotting utilities for the DDSP library. Useful in colab and elsewhere."""

from ddsp import core
from ddsp import spectral_ops
from matplotlib import gridspec
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


def pianoroll_plot_setup(figsize=None, side_piano_ratio=0.025,
                         faint_pr=True, xlim=None):
  """Makes a tiny piano left of the y-axis and a faint piano on the main figure.

  This function sets up the figure for pretty plotting a piano roll. It makes a
  small imshow plot to the left of the main plot that looks like a piano. This
  piano side plot is aligned along the y-axis of the main plot, such that y
  values align with MIDI values (y=0 is the lowest C-1, y=11 is C0, etc).
  Additionally, a main figure is set up that shares the y-axis of the piano side
  plot. Optionally, a set of faint horizontal lines are drawn on the main figure
  that correspond to the black keys on the piano (and a line separating B & C
  and E & F). This function returns the formatted figure, the side piano axis,
  and the main axis for plotting your data.

  By default, this will draw 11 octaves of piano keys along the y-axis; you will
  probably want reduce what is visible using `ax.set_ylim()` on either returned
  axis.

  Using with imshow piano roll data:
    A common use case is for using imshow() on the main axis to display a piano
    roll alongside the piano side plot AND the faint piano roll behind your
    data. In this case, if your data is a 2D array you have to use a masked
    numpy array to make certain values invisible on the plot, and therefore make
    the faint piano roll visible. Here's an example:

    midi = np.flipud([
          [0.0, 0.0, 1.0],
          [0.0, 1.0, 0.0],
          [1.0, 0.0, 0.0],
    ])

    midi_masked = np.ma.masked_values(midi, 0.0)  # Mask out all 0.0's
    fig, ax, sp = plotting.pianoroll_plot_setup()
    ax.imshow(midi_masked, origin='lower', aspect='auto')  # main subplot axis
    sp.set_ylabel('My favorite MIDI data')  # side piano axis
    fig.show()

    The other option is to use imshow in RGBA mode, where your data is split
    into 4 channels. Every alpha value that is 0.0 will be transparent and show
    the faint piano roll below your data.

  Args:
    figsize: Size if the matplotlib figure. Will be passed to `plt.figure()`.
      Defaults to None.
    side_piano_ratio: Width of the y-axis piano in terms of raio of the whole
      figure. Defaults to 1/40th.
    faint_pr: Whether to draw faint black & white keys across the main plot.
      Defaults to True.
    xlim: Tuple containing the min and max of the x values for the main plot.
      Only used to determine the x limits for the faint piano roll in the main
      plot. Defaults to (0, 1000).

  Returns:
    (figure, main_axis, left_piano_axis)
      figure: A matplotlib figure object containing both subplots set up with an
        aligned piano roll.
      main_axis: A matplotlib axis object to be used for plotting. Optionally
        has a faint piano roll in the background.
      left_piano_axis: A matplotlib axis object that has a small, aligned piano
        along the left side y-axis of the main_axis subplot.
  """
  octaves = 11

  # Setup figure and gridspec.
  fig = plt.figure(figsize=figsize)
  gs_ratio = int(1 / side_piano_ratio)
  gs = gridspec.GridSpec(1, 2, width_ratios=[1, gs_ratio])
  left_piano_ax = fig.add_subplot(gs[0])

  # Make a piano on the left side of the y-axis with imshow().
  keys = np.array(
      [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0]  # notes in descending order; B -> C
  )
  keys = np.tile(keys, octaves)[:, None]
  left_piano_ax.imshow(keys, cmap='binary', aspect='auto',
                       extent=[0, 0.625, -0.5, octaves*12-0.5])

  # Make the lines between keys.
  for i in range(octaves):
    left_piano_ax.hlines(i*12 - 0.5, -0.5, 1, colors='black', linewidth=0.5)
    left_piano_ax.hlines(i*12 + 1.0, -0.5, 1, colors='black', linewidth=0.5)
    left_piano_ax.hlines(i*12 + 3.0, -0.5, 1, colors='black', linewidth=0.5)
    left_piano_ax.hlines(i*12 + 4.5, -0.5, 1, colors='black', linewidth=0.5)
    left_piano_ax.hlines(i*12 + 6.0, -0.5, 1, colors='black', linewidth=0.5)
    left_piano_ax.hlines(i*12 + 8.0, -0.5, 1, colors='black', linewidth=0.5)
    left_piano_ax.hlines(i*12 + 10.0, -0.5, 1, colors='black', linewidth=0.5)

  # Set the limits of the side piano and remove ticks so it looks nice.
  left_piano_ax.set_xlim(0, 0.995)
  left_piano_ax.set_xticks([])

  # Create the aligned axis we'll return to the user.
  main_ax = fig.add_subplot(gs[1], sharey=left_piano_ax)

  # Draw a faint piano roll behind the main axes (if the user wants).
  if faint_pr:
    xlim = (0, 1000) if xlim is None else xlim
    x_min, x_max = xlim
    x_delta = x_max - x_min
    main_ax.imshow(np.tile(keys, x_delta), cmap='binary', aspect='auto',
                   alpha=0.05, extent=[x_min, x_max, -0.5, octaves*12-0.5])
    for i in range(octaves):
      main_ax.hlines(i * 12 + 4.5, x_min, x_max, colors='black',
                     linewidth=0.5, alpha=0.25)
      main_ax.hlines(i * 12 - 0.5, x_min, x_max, colors='black',
                     linewidth=0.5, alpha=0.25)

    main_ax.set_xlim(*xlim)

  # Some final cosmetic tweaks before returning the axis obj's and figure.
  plt.setp(main_ax.get_yticklabels(), visible=False)
  gs.tight_layout(fig)
  return fig, main_ax, left_piano_ax
