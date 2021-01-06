# Copyright 2021 The DDSP Authors.
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

import base64
import io
import pickle

import ddsp
import ddsp.training
from IPython import display
import note_seq
import numpy as np
from scipy import stats
from scipy.io import wavfile
import tensorflow.compat.v2 as tf

from google.colab import files
from google.colab import output

download = files.download

DEFAULT_SAMPLE_RATE = ddsp.spectral_ops.CREPE_SAMPLE_RATE

_play_count = 0  # Used for ephemeral play().

# Alias these for backwards compatibility and ease.
specplot = ddsp.training.plotting.specplot
plot_impulse_responses = ddsp.training.plotting.plot_impulse_responses
transfer_function = ddsp.training.plotting.transfer_function


# ------------------------------------------------------------------------------
# IO
# ------------------------------------------------------------------------------
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
  # If batched, take first element.
  if len(array_of_floats.shape) == 2:
    array_of_floats = array_of_floats[0]

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
  global _play_count
  _play_count += 1
  if ephemeral:
    element = 'id_%s' % _play_count
    display.display(display.HTML('<div id="%s"> </div>' % element))
    js = output._js_builder  # pylint:disable=protected-access
    js.Js('document', mode=js.EVAL).getElementById(element).innerHTML = html
  else:
    display.display(display.HTML(html))


def record(seconds=3, sample_rate=DEFAULT_SAMPLE_RATE, normalize_db=0.1):
  """Record audio from the browser in colab using javascript.

  Based on: https://gist.github.com/korakot/c21c3476c024ad6d56d5f48b0bca92be
  Args:
    seconds: Number of seconds to record.
    sample_rate: Resample recorded audio to this sample rate.
    normalize_db: Normalize the audio to this many decibels. Set to None to skip
      normalization step.

  Returns:
    An array of the recorded audio at sample_rate.
  """
  # Use Javascript to record audio.
  record_js_code = """
  const sleep  = time => new Promise(resolve => setTimeout(resolve, time))
  const b2text = blob => new Promise(resolve => {
    const reader = new FileReader()
    reader.onloadend = e => resolve(e.srcElement.result)
    reader.readAsDataURL(blob)
  })

  var record = time => new Promise(async resolve => {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    recorder = new MediaRecorder(stream)
    chunks = []
    recorder.ondataavailable = e => chunks.push(e.data)
    recorder.start()
    await sleep(time)
    recorder.onstop = async ()=>{
      blob = new Blob(chunks)
      text = await b2text(blob)
      resolve(text)
    }
    recorder.stop()
  })
  """
  print('Starting recording for {} seconds...'.format(seconds))
  display.display(display.Javascript(record_js_code))
  audio_string = output.eval_js('record(%d)' % (seconds * 1000.0))
  print('Finished recording!')
  audio_bytes = base64.b64decode(audio_string.split(',')[1])
  return audio_bytes_to_np(audio_bytes,
                           sample_rate=sample_rate,
                           normalize_db=normalize_db)


def audio_bytes_to_np(wav_data,
                      sample_rate=DEFAULT_SAMPLE_RATE,
                      normalize_db=0.1):
  """Convert audio file data (in bytes) into a numpy array using Pydub.

  Args:
    wav_data: A byte stream of audio data.
    sample_rate: Resample recorded audio to this sample rate.
    normalize_db: Normalize the audio to this many decibels. Set to None to skip
      normalization step.

  Returns:
    An array of the recorded audio at sample_rate.
  """
  return note_seq.audio_io.wav_data_to_samples_pydub(
      wav_data=wav_data, sample_rate=sample_rate, normalize_db=normalize_db)


def upload(sample_rate=DEFAULT_SAMPLE_RATE, normalize_db=None):
  """Load a collection of audio files (.wav, .mp3) from disk into colab.

  Args:
    sample_rate: Resample recorded audio to this sample rate.
    normalize_db: Normalize the audio to this many decibels. Set to None to skip
      normalization step.

  Returns:
    An tuple of lists, (filenames, numpy_arrays).
  """
  audio_files = files.upload()
  fnames = list(audio_files.keys())
  audio = []
  for fname in fnames:
    file_audio = audio_bytes_to_np(audio_files[fname],
                                   sample_rate=sample_rate,
                                   normalize_db=normalize_db)
    audio.append(file_audio)
  return fnames, audio


# ------------------------------------------------------------------------------
# Loudness Normalization
# ------------------------------------------------------------------------------
def smooth(x, filter_size=3):
  """Smooth 1-d signal with a box filter."""
  x = tf.convert_to_tensor(x, tf.float32)
  is_2d = len(x.shape) == 2
  x = x[:, :, tf.newaxis] if is_2d else x[tf.newaxis, :, tf.newaxis]
  w = tf.ones([filter_size])[:, tf.newaxis, tf.newaxis] / float(filter_size)
  y = tf.nn.conv1d(x, w, stride=1, padding='SAME')
  y = y[:, :, 0] if is_2d else y[0, :, 0]
  return y.numpy()


def detect_notes(loudness_db,
                 f0_confidence,
                 note_threshold=1.0,
                 exponent=2.0,
                 smoothing=40,
                 f0_confidence_threshold=0.7,
                 min_db=-120.):
  """Detect note on-off using loudness and smoothed f0_confidence."""
  mean_db = np.mean(loudness_db)
  db = smooth(f0_confidence**exponent, smoothing) * (loudness_db - min_db)
  db_threshold = (mean_db - min_db) * f0_confidence_threshold**exponent
  note_on_ratio = db / db_threshold
  mask_on = note_on_ratio >= note_threshold
  return mask_on, note_on_ratio


class QuantileTransformer:
  """Transform features using quantiles information.

  Stripped down version of sklearn.preprocessing.QuantileTransformer.
  https://github.com/scikit-learn/scikit-learn/blob/
  863e58fcd5ce960b4af60362b44d4f33f08c0f97/sklearn/preprocessing/_data.py

  Putting directly in ddsp library to avoid dependency on sklearn that breaks
  when pickling and unpickling from different versions of sklearn.
  """

  def __init__(self,
               n_quantiles=1000,
               output_distribution='uniform',
               subsample=int(1e5)):
    """Constructor.

    Args:
      n_quantiles: int, default=1000 or n_samples Number of quantiles to be
        computed. It corresponds to the number of landmarks used to discretize
        the cumulative distribution function. If n_quantiles is larger than the
        number of samples, n_quantiles is set to the number of samples as a
        larger number of quantiles does not give a better approximation of the
        cumulative distribution function estimator.
      output_distribution: {'uniform', 'normal'}, default='uniform' Marginal
        distribution for the transformed data. The choices are 'uniform'
        (default) or 'normal'.
      subsample: int, default=1e5 Maximum number of samples used to estimate
        the quantiles for computational efficiency. Note that the subsampling
        procedure may differ for value-identical sparse and dense matrices.
    """
    self.n_quantiles = n_quantiles
    self.output_distribution = output_distribution
    self.subsample = subsample
    self.random_state = np.random.mtrand._rand

  def _dense_fit(self, x, random_state):
    """Compute percentiles for dense matrices.

    Args:
      x: ndarray of shape (n_samples, n_features)
        The data used to scale along the features axis.
      random_state: Numpy random number generator.
    """
    n_samples, _ = x.shape
    references = self.references_ * 100

    self.quantiles_ = []
    for col in x.T:
      if self.subsample < n_samples:
        subsample_idx = random_state.choice(
            n_samples, size=self.subsample, replace=False)
        col = col.take(subsample_idx, mode='clip')
      self.quantiles_.append(np.nanpercentile(col, references))
    self.quantiles_ = np.transpose(self.quantiles_)
    # Due to floating-point precision error in `np.nanpercentile`,
    # make sure that quantiles are monotonically increasing.
    # Upstream issue in numpy:
    # https://github.com/numpy/numpy/issues/14685
    self.quantiles_ = np.maximum.accumulate(self.quantiles_)

  def fit(self, x):
    """Compute the quantiles used for transforming.

    Parameters
    ----------
    Args:
      x: {array-like, sparse matrix} of shape (n_samples, n_features)
        The data used to scale along the features axis. If a sparse
        matrix is provided, it will be converted into a sparse
        ``csc_matrix``. Additionally, the sparse matrix needs to be
        nonnegative if `ignore_implicit_zeros` is False.

    Returns:
      self: object
         Fitted transformer.
    """
    if self.n_quantiles <= 0:
      raise ValueError("Invalid value for 'n_quantiles': %d. "
                       'The number of quantiles must be at least one.' %
                       self.n_quantiles)
    n_samples = x.shape[0]
    self.n_quantiles_ = max(1, min(self.n_quantiles, n_samples))

    # Create the quantiles of reference
    self.references_ = np.linspace(0, 1, self.n_quantiles_, endpoint=True)
    self._dense_fit(x, self.random_state)
    return self

  def _transform_col(self, x_col, quantiles, inverse):
    """Private function to transform a single feature."""
    output_distribution = self.output_distribution
    bounds_threshold = 1e-7

    if not inverse:
      lower_bound_x = quantiles[0]
      upper_bound_x = quantiles[-1]
      lower_bound_y = 0
      upper_bound_y = 1
    else:
      lower_bound_x = 0
      upper_bound_x = 1
      lower_bound_y = quantiles[0]
      upper_bound_y = quantiles[-1]
      # for inverse transform, match a uniform distribution
      with np.errstate(invalid='ignore'):  # hide NaN comparison warnings
        if output_distribution == 'normal':
          x_col = stats.norm.cdf(x_col)
        # else output distribution is already a uniform distribution

    # find index for lower and higher bounds
    with np.errstate(invalid='ignore'):  # hide NaN comparison warnings
      if output_distribution == 'normal':
        lower_bounds_idx = (x_col - bounds_threshold < lower_bound_x)
        upper_bounds_idx = (x_col + bounds_threshold > upper_bound_x)
      if output_distribution == 'uniform':
        lower_bounds_idx = (x_col == lower_bound_x)
        upper_bounds_idx = (x_col == upper_bound_x)

    isfinite_mask = ~np.isnan(x_col)
    x_col_finite = x_col[isfinite_mask]
    if not inverse:
      # Interpolate in one direction and in the other and take the
      # mean. This is in case of repeated values in the features
      # and hence repeated quantiles
      #
      # If we don't do this, only one extreme of the duplicated is
      # used (the upper when we do ascending, and the
      # lower for descending). We take the mean of these two
      x_col[isfinite_mask] = .5 * (
          np.interp(x_col_finite, quantiles, self.references_) -
          np.interp(-x_col_finite, -quantiles[::-1], -self.references_[::-1]))
    else:
      x_col[isfinite_mask] = np.interp(x_col_finite, self.references_,
                                       quantiles)

    x_col[upper_bounds_idx] = upper_bound_y
    x_col[lower_bounds_idx] = lower_bound_y
    # for forward transform, match the output distribution
    if not inverse:
      with np.errstate(invalid='ignore'):  # hide NaN comparison warnings
        if output_distribution == 'normal':
          x_col = stats.norm.ppf(x_col)
          # find the value to clip the data to avoid mapping to
          # infinity. Clip such that the inverse transform will be
          # consistent
          clip_min = stats.norm.ppf(bounds_threshold - np.spacing(1))
          clip_max = stats.norm.ppf(1 - (bounds_threshold - np.spacing(1)))
          x_col = np.clip(x_col, clip_min, clip_max)
        # else output distribution is uniform and the ppf is the
        # identity function so we let x_col unchanged

    return x_col

  def _transform(self, x, inverse=False):
    """Forward and inverse transform.

    Args:
      x : ndarray of shape (n_samples, n_features)
        The data used to scale along the features axis.
      inverse : bool, default=False
        If False, apply forward transform. If True, apply
        inverse transform.

    Returns:
      x : ndarray of shape (n_samples, n_features)
        Projected data
    """
    x = np.array(x)  # Explicit copy.
    for feature_idx in range(x.shape[1]):
      x[:, feature_idx] = self._transform_col(
          x[:, feature_idx], self.quantiles_[:, feature_idx], inverse)
    return x

  def transform(self, x):
    """Feature-wise transformation of the data."""
    return self._transform(x, inverse=False)

  def inverse_transform(self, x):
    """Back-projection to the original space."""
    return self._transform(x, inverse=True)

  def fit_transform(self, x):
    """Fit and transform."""
    return self.fit(x).transform(x)


def fit_quantile_transform(loudness_db, mask_on, inv_quantile=None):
  """Fits quantile normalization, given a note_on mask.

  Optionally, performs the inverse transformation given a pre-fitted transform.
  Args:
    loudness_db: Decibels, shape [batch, time]
    mask_on: A binary mask for when a note is present, shape [batch, time].
    inv_quantile: Optional pretrained QuantileTransformer to perform the inverse
      transformation.

  Returns:
    Trained quantile transform. Also returns the renormalized loudnesses if
      inv_quantile is provided.
  """
  quantile_transform = QuantileTransformer()
  loudness_flat = np.ravel(loudness_db[mask_on])[:, np.newaxis]
  loudness_flat_q = quantile_transform.fit_transform(loudness_flat)

  if inv_quantile is None:
    return quantile_transform
  else:
    loudness_flat_norm = inv_quantile.inverse_transform(loudness_flat_q)
    loudness_norm = np.ravel(loudness_db.copy())[:, np.newaxis]
    loudness_norm[mask_on] = loudness_flat_norm
    return quantile_transform, loudness_norm


def save_dataset_statistics(data_provider, file_path, batch_size=1):
  """Calculate dataset stats and save in a pickle file."""
  print('Calculating dataset statistics for', data_provider)
  data_iter = iter(data_provider.get_batch(batch_size, repeats=1))

  # Unpack dataset.
  i = 0
  loudness = []
  f0 = []
  f0_conf = []
  audio = []

  for batch in data_iter:
    loudness.append(batch['loudness_db'])
    f0.append(batch['f0_hz'])
    f0_conf.append(batch['f0_confidence'])
    audio.append(batch['audio'])
    i += 1

  print(f'Computing statistics for {i * batch_size} examples.')

  loudness = np.vstack(loudness)
  f0 = np.vstack(f0)
  f0_conf = np.vstack(f0_conf)
  audio = np.vstack(audio)

  # Fit the transform.
  trim_end = 20
  f0_trimmed = f0[:, :-trim_end]
  l_trimmed = loudness[:, :-trim_end]
  f0_conf_trimmed = f0_conf[:, :-trim_end]
  mask_on, _ = detect_notes(l_trimmed, f0_conf_trimmed)
  quantile_transform = fit_quantile_transform(l_trimmed, mask_on)

  # Average values.
  mean_pitch = np.mean(ddsp.core.hz_to_midi(f0_trimmed[mask_on]))
  mean_loudness = np.mean(l_trimmed)
  mean_max_loudness = np.mean(np.max(l_trimmed, axis=0))

  # Object to pickle all the statistics together.
  ds = {'mean_pitch': mean_pitch,
        'mean_loudness': mean_loudness,
        'mean_max_loudness': mean_max_loudness,
        'quantile_transform': quantile_transform}

  # Save.
  with tf.io.gfile.GFile(file_path, 'wb') as f:
    pickle.dump(ds, f)
  print(f'Done! Saved to: {file_path}')


# ------------------------------------------------------------------------------
# Frequency tuning
# ------------------------------------------------------------------------------
def get_tuning_factor(f0_midi, f0_confidence, mask_on):
  """Get an offset in cents, to most consistent set of chromatic intervals."""
  # Difference from midi offset by different tuning_factors.
  tuning_factors = np.linspace(-0.5, 0.5, 101)  # 1 cent divisions.
  midi_diffs = (f0_midi[mask_on][:, np.newaxis] -
                tuning_factors[np.newaxis, :]) % 1.0
  midi_diffs[midi_diffs > 0.5] -= 1.0
  weights = f0_confidence[mask_on][:, np.newaxis]

  ## Computes mininmum adjustment distance.
  cost_diffs = np.abs(midi_diffs)
  cost_diffs = np.mean(weights * cost_diffs, axis=0)

  ## Computes mininmum "note" transitions.
  f0_at = f0_midi[mask_on][:, np.newaxis] - midi_diffs
  f0_at_diffs = np.diff(f0_at, axis=0)
  deltas = (f0_at_diffs != 0.0).astype(np.float)
  cost_deltas = np.mean(weights[:-1] * deltas, axis=0)

  # Tuning factor is minimum cost.
  norm = lambda x: (x - np.mean(x)) / np.std(x)
  cost = norm(cost_deltas) + norm(cost_diffs)
  return tuning_factors[np.argmin(cost)]


def auto_tune(f0_midi, tuning_factor, mask_on, amount=0.0, chromatic=False):
  """Reduce variance of f0 from the chromatic or scale intervals."""
  if chromatic:
    midi_diff = (f0_midi - tuning_factor) % 1.0
    midi_diff[midi_diff > 0.5] -= 1.0
  else:
    major_scale = np.ravel(
        [np.array([0, 2, 4, 5, 7, 9, 11]) + 12 * i for i in range(10)])
    all_scales = np.stack([major_scale + i for i in range(12)])

    f0_on = f0_midi[mask_on]
    # [time, scale, note]
    f0_diff_tsn = (
        f0_on[:, np.newaxis, np.newaxis] - all_scales[np.newaxis, :, :])
    # [time, scale]
    f0_diff_ts = np.min(np.abs(f0_diff_tsn), axis=-1)
    # [scale]
    f0_diff_s = np.mean(f0_diff_ts, axis=0)
    scale_idx = np.argmin(f0_diff_s)
    scale = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb',
             'G', 'Ab', 'A', 'Bb', 'B', 'C'][scale_idx]

    # [time]
    f0_diff_tn = f0_midi[:, np.newaxis] - all_scales[scale_idx][np.newaxis, :]
    note_idx = np.argmin(np.abs(f0_diff_tn), axis=-1)
    midi_diff = np.take_along_axis(
        f0_diff_tn, note_idx[:, np.newaxis], axis=-1)[:, 0]
    print('Autotuning... \nInferred key: {}  '
          '\nTuning offset: {} cents'.format(scale, int(tuning_factor * 100)))

  # Adjust the midi signal.
  return f0_midi - amount * midi_diff
