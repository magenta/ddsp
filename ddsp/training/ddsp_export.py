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

r"""Convert checkpoint to SavedModel, and SavedModel to TFJS / TFLite format.

Example Usage (Defaults):
ddsp_export --model_path=/path/to/model

Example Usage (TFJS model):
ddsp_export --model_path=/path/to/model --inference_model=autoencoder \
--tflite=false --tfjs

Example Usage (TFLite model):
ddsp_export --model_path=/path/to/model --inference_model=streaming_f0_pw \
--tflite --tfjs=false

Example Usage (SavedModel Only):
ddsp_export --model_path=/path/to/model --inference_model=[model_type] \
--tflite=false --tfjs=false
"""

import datetime
import json
import os

from absl import app
from absl import flags

import ddsp
from ddsp.training import data
from ddsp.training import inference
from ddsp.training import postprocessing
from ddsp.training import train_util
import gin
import librosa
import note_seq
import tensorflow as tf
from tensorflowjs.converters import converter

# pylint: disable=pointless-string-statement

from tflite_support import metadata as _metadata
# pylint: enable=pointless-string-statement

flags.DEFINE_string(
    'name', '', 'Name of your model to use as folder and filename on export. '
    'Defaults to "export/" and "model.tflite" if none is provided.')
flags.DEFINE_string(
    'model_path', '', 'Path to checkpoint or SavedModel directory. If no '
    'SavedModel is found, will search for latest checkpoint '
    'use it to create a SavedModel. Can also provide direct '
    'path to desired checkpoint. E.g. `/path/to/ckpt-[iter]`.')
flags.DEFINE_string(
    'save_dir', '', 'Optional directory in which to save converted checkpoint.'
    'If none is provided, it will be FLAGS.model_path if it '
    'contains a SavedModel, otherwise FLAGS.model_path/export.')

# Specify model class.
flags.DEFINE_enum(
    'inference_model',
    'streaming_f0_pw',
    [
        'autoencoder',
        'streaming_f0_pw',
        'vst_extract_features',
        'vst_predict_controls',
        'vst_stateless_predict_controls',
        'vst_synthesize',
        'vst_synthesize_harmonic',
        'vst_synthesize_noise',
    ],
    'Specify the ddsp.training.inference model to use for '
    'converting a checkpoint to a SavedModel. Names are '
    'snake_case versions of class names.')

# Optional flags.
flags.DEFINE_multi_string('gin_param', [],
                          'Gin parameters for custom inference model kwargs.')
flags.DEFINE_boolean('debug', False, 'DEBUG: Do not save the model')

# Conversion formats.
flags.DEFINE_boolean('tfjs', True,
                     'Convert SavedModel to TFJS for deploying on the web.')
flags.DEFINE_boolean('tflite', True,
                     'Convert SavedModel to TFLite for embedded C++ apps.')
flags.DEFINE_string('metadata_file', None,
                    'Optional metadata file to pack into TFLite model.')

FLAGS = flags.FLAGS

# Metadata.
flags.DEFINE_boolean('metadata', True, 'Save metadata for model as a json.')
flags.DEFINE_string(
    'dataset_path', None,
    'Only required if FLAGS.metadata=True. Path to TF Records containing '
    'training examples. Only used if no binding to train.data_provider can '
    'be found.')

# Reverb Impulse Response.
flags.DEFINE_boolean('reverb', True,
                     'Save reverb impulse response as a wav file.')
flags.DEFINE_integer('reverb_sample_rate', 44100,
                     'If not None, also save resampled reverb ir.')

FLAGS = flags.FLAGS


def get_data_provider(dataset_path, model_path):
  """Get the data provider for dataset for statistics.

  Read TF examples from specified path if provided, else use the
  data provider specified in the gin config.
  Args:
    dataset_path: Path to an sstable of TF Examples.
    model_path: Path to the model checkpoint dir containing the gin config.
  Returns:
    Data provider to calculate statistics over.
  """
  # First, see if the dataset path is specified
  if dataset_path is not None:
    dataset_path = train_util.expand_path(dataset_path)
    return data.TFRecordProvider(dataset_path)
  else:
    inference.parse_operative_config(model_path)
    try:
      dp_binding = gin.query_parameter('train.data_provider')
      return dp_binding.scoped_configurable_fn()

    except ValueError as e:
      raise Exception(
          'Failed to parse dataset from gin. Either --dataset_path '
          'or train.data_provider gin param must be set.') from e


def get_metadata_dict(data_provider, model_path):
  """Compute metadata using compute_dataset_statistics and add version/date."""

  # Parse gin for num_harmonics and num_noise_amps.
  inference.parse_operative_config(model_path)

  # Get number of outputs.
  ref = gin.query_parameter('Autoencoder.decoder')
  decoder_type = ref.config_key[-1].split('.')[-1]
  output_splits = dict(gin.query_parameter(f'{decoder_type}.output_splits'))

  # Get power rate and size.
  frame_size = gin.query_parameter('%frame_size')
  frame_rate = gin.query_parameter('%frame_rate')
  sample_rate = gin.query_parameter('%sample_rate')

  # Compute stats.
  full_metadata = postprocessing.compute_dataset_statistics(
      data_provider,
      power_frame_size=frame_size,
      power_frame_rate=frame_rate)

  lite_metadata = {
      'mean_min_pitch_note':
          float(full_metadata['mean_min_pitch_note']),
      'mean_max_pitch_note':
          float(full_metadata['mean_max_pitch_note']),
      'mean_min_pitch_note_hz':
          float(ddsp.core.midi_to_hz(full_metadata['mean_min_pitch_note'])),
      'mean_max_pitch_note_hz':
          float(ddsp.core.midi_to_hz(full_metadata['mean_max_pitch_note'])),
      'mean_min_power_note':
          float(full_metadata['mean_min_power_note']),
      'mean_max_power_note':
          float(full_metadata['mean_max_power_note']),
      'version':
          ddsp.__version__,
      'export_time':
          datetime.datetime.now().isoformat(),
      'num_harmonics':
          output_splits['harmonic_distribution'],
      'num_noise_amps':
          output_splits['noise_magnitudes'],
      'frame_rate':
          frame_rate,
      'frame_size':
          frame_size,
      'sample_rate':
          sample_rate,
  }
  return lite_metadata


def get_inference_model(ckpt):
  """Restore model from checkpoint using global FLAGS.

  Use --gin_param for any custom kwargs for model constructors.
  Args:
    ckpt: Path to the checkpoint.

  Returns:
    Inference model, built and restored from checkpoint.
  """
  # Parse model kwargs from --gin_param.
  print('Parsing --gin_param flags:', FLAGS.gin_param)
  with gin.unlock_config():
    gin.parse_config_files_and_bindings(None, FLAGS.gin_param)

  models = {
      'autoencoder': inference.AutoencoderInference,
      'vst_extract_features': inference.VSTExtractFeatures,
      'vst_predict_controls': inference.VSTPredictControls,
      'vst_stateless_predict_controls': inference.VSTStatelessPredictControls,
      'vst_synthesize': inference.VSTSynthesize,
      'vst_synthesize_harmonic': inference.VSTSynthesizeHarmonic,
      'vst_synthesize_noise': inference.VSTSynthesizeNoise,
  }
  return models[FLAGS.inference_model](ckpt)


def ckpt_to_saved_model(ckpt, save_dir):
  """Convert Checkpoint to SavedModel."""
  print(f'\nConverting to SavedModel:' f'\nInput: {ckpt}\nOutput: {save_dir}\n')
  model = get_inference_model(ckpt)
  print('Finshed Loading Model!')
  if not FLAGS.debug:
    model.save_model(save_dir)
  print('SavedModel Conversion Success!')


def saved_model_to_tfjs(input_dir, save_dir):
  """Convert SavedModel to TFJS model."""
  print(f'\nConverting to TFJS:\nInput:{input_dir}\nOutput:{save_dir}\n')
  converter.convert([
      '--input_format=tf_saved_model', '--signature_name=serving_default',
      '--control_flow_v2=True', '--skip_op_check', '--quantize_float16=True',
      '--experiments=True', input_dir, save_dir
  ])
  print('TFJS Conversion Success!')


def saved_model_to_tflite(input_dir,
                          save_dir,
                          metadata_file=None,
                          name=''):
  """Convert SavedModel to TFLite model."""
  print(f'\nConverting to TFLite:\nInput:{input_dir}\nOutput:{save_dir}\n')
  # Convert the model.
  tflite_converter = tf.lite.TFLiteConverter.from_saved_model(input_dir)
  tflite_converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops.
      tf.lite.OpsSet.SELECT_TF_OPS,  # Enable extended TensorFlow ops.
  ]
  tflite_model = tflite_converter.convert()  # Byte string.
  # Save the model.
  name = name if name else 'model'
  save_path = os.path.join(save_dir, f'{name}.tflite')
  with tf.io.gfile.GFile(save_path, 'wb') as f:
    f.write(tflite_model)

  if metadata_file is not None:
    populator = _metadata.MetadataPopulator.with_model_file(save_path)
    populator.load_associated_files([metadata_file])
    populator.populate()
  print('TFLite Conversion Success!')


def export_impulse_response(model_path, save_dir, target_sr=None):
  """Extracts and saves the reverb impulse response."""
  with gin.unlock_config():
    ddsp.training.inference.parse_operative_config(model_path)
    model = ddsp.training.models.Autoencoder()
    model.restore(model_path)
  sr = model.processor_group.harmonic.sample_rate
  reverb = model.processor_group.reverb
  reverb.build(unused_input_shape=[])
  ir = reverb.get_controls(audio=tf.zeros([1, 1]))['ir'].numpy()[0]
  print(f'Reverb Impulse Response is {ir.shape[0] / sr} seconds long')

  def save_ir(ir, sr):
    """Save the impulse response."""
    ir_path = os.path.join(save_dir, f'reverb_ir_{sr}_hz.wav')
    with tf.io.gfile.GFile(ir_path, 'wb') as f:
      wav_data = note_seq.audio_io.samples_to_wav_data(ir, sr)
      f.write(wav_data)

  # Save to original impulse response.
  save_ir(ir, sr)

  # Save the resampled impulse response.
  if target_sr is not None:
    sr = target_sr
    ir = librosa.resample(ir, orig_sr=sr, target_sr=target_sr)
    save_ir(ir, sr)


def ensure_exits(dir_path):
  """Make directory if none exists."""
  if not tf.io.gfile.exists(dir_path):
    tf.io.gfile.makedirs(dir_path)


def main(unused_argv):
  model_path = train_util.expand_path(FLAGS.model_path)

  # Figure out what type the model path is.
  is_saved_model = tf.io.gfile.exists(
      os.path.join(model_path, 'saved_model.pb'))
  is_ckpt = not tf.io.gfile.isdir(model_path)

  # Infer save directory path.
  export_name = FLAGS.name if FLAGS.name else 'export'
  if FLAGS.save_dir:
    save_dir = FLAGS.save_dir
  else:
    if is_saved_model:
      # If model_path is a SavedModel, use that directory.
      save_dir = model_path
    elif is_ckpt:
      # If model_path is a checkpoint file, use the directory of the file.
      save_dir = os.path.join(os.path.dirname(model_path), export_name)
    else:
      # If model_path is a checkpoint directory, use child export directory.
      save_dir = os.path.join(model_path, export_name)

  # Make a new save directory.
  save_dir = train_util.expand_path(save_dir)
  ensure_exits(save_dir)

  # Save reverb impulse response.
  if FLAGS.reverb:
    export_impulse_response(model_path, save_dir, FLAGS.reverb_sample_rate)

  # Save metadata.
  if FLAGS.metadata:
    metadata_path = os.path.join(save_dir, 'metadata.json')
    data_provider = get_data_provider(FLAGS.dataset_path, model_path)
    metadata = get_metadata_dict(data_provider, model_path)
    with tf.io.gfile.GFile(metadata_path, 'w') as f:
      f.write(json.dumps(metadata))

  # Create SavedModel if none already exists.
  if not is_saved_model:
    ckpt_to_saved_model(model_path, save_dir)

  # Convert SavedModel.
  if FLAGS.tfjs:
    tfjs_dir = os.path.join(save_dir, 'tfjs')
    ensure_exits(tfjs_dir)
    saved_model_to_tfjs(save_dir, tfjs_dir)

  if FLAGS.tflite:
    tflite_dir = os.path.join(save_dir, 'tflite')
    ensure_exits(tflite_dir)
    saved_model_to_tflite(save_dir,
                          tflite_dir,
                          metadata_path if FLAGS.metadata else '',
                          name=FLAGS.name)


def console_entry_point():
  """From pip installed script."""
  app.run(main)


if __name__ == '__main__':
  console_entry_point()
