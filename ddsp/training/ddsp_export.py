# Copyright 2022 The DDSP Authors.
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

import os

from absl import app
from absl import flags

from ddsp.training import inference
from ddsp.training import train_util
import gin
import tensorflow as tf
from tensorflowjs.converters import converter


from tflite_support import metadata as _metadata


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


def saved_model_to_tflite(input_dir, save_dir, metadata_file=None):
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
  save_path = os.path.join(save_dir, 'model.tflite')
  with tf.io.gfile.GFile(save_path, 'wb') as f:
    f.write(tflite_model)

  if metadata_file is not None:
    populator = _metadata.MetadataPopulator.with_model_file(save_path)
    populator.load_associated_files([metadata_file])
    populator.populate()
  print('TFLite Conversion Success!')


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
  if FLAGS.save_dir:
    save_dir = FLAGS.save_dir
  else:
    if is_saved_model:
      # If model_path is a SavedModel, use that directory.
      save_dir = model_path
    elif is_ckpt:
      # If model_path is a checkpoint file, use the directory of the file.
      save_dir = os.path.join(os.path.dirname(model_path), 'export')
    else:
      # If model_path is a checkpoint directory, use child export directory.
      save_dir = os.path.join(model_path, 'export')

  # Make a new save directory.
  save_dir = train_util.expand_path(save_dir)
  ensure_exits(save_dir)

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
    saved_model_to_tflite(save_dir, tflite_dir, FLAGS.metadata_file)


def console_entry_point():
  """From pip installed script."""
  app.run(main)


if __name__ == '__main__':
  console_entry_point()
