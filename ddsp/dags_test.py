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

"""Tests for ddsp.dags.py."""

from absl.testing import parameterized
from ddsp import dags
import gin
import tensorflow as tf

# Make dense layers configurable for this test.
gin.external_configurable(tf.keras.layers.Dense, 'tf.keras.layers.Dense')


@gin.configurable
class ConfigurableDAGLayer(dags.DAGLayer):
  """Configurable wrapper DAGLayer encapsulated for this test."""
  pass


class DAGLayerTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    """Create some dummy input data for the chain."""
    super().setUp()
    # Create inputs.
    self.n_batch = 4
    self.x_dims = 5
    self.z_dims = 2
    self.x = tf.ones([self.n_batch, self.x_dims])
    self.inputs = {'test_data': self.x}
    self.gin_config_kwarg_modules = f"""
    import ddsp

    ### Modules
    ConfigurableDAGLayer.dag = [
        ('encoder', ['inputs/test_data'], ['z']),
        ('bottleneck', ['encoder/z'], ['z_bottleneck']),
        ('decoder', ['bottleneck/z_bottleneck'], ['reconstruction']),
    ]
    ConfigurableDAGLayer.encoder = @encoder/layers.Dense()
    encoder/layers.Dense.units = {self.x_dims}

    ConfigurableDAGLayer.bottleneck = @bottleneck/layers.Dense()
    bottleneck/layers.Dense.units = {self.z_dims}

    ConfigurableDAGLayer.decoder = @decoder/layers.Dense()
    decoder/layers.Dense.units = {self.x_dims}
    """
    self.gin_config_dag_modules = f"""
    import ddsp

    ### Modules
    ConfigurableDAGLayer.dag = [
        (@encoder/layers.Dense(), ['inputs/test_data'], ['z']),
        (@bottleneck/layers.Dense(), ['encoder/z'], ['z_bottleneck']),
        (@decoder/layers.Dense(), ['bottleneck/z_bottleneck'], ['reconstruction']),
    ]
    encoder/layers.Dense.name = 'encoder'
    encoder/layers.Dense.units = {self.x_dims}

    bottleneck/layers.Dense.name = 'bottleneck'
    bottleneck/layers.Dense.units = {self.z_dims}

    decoder/layers.Dense.name = 'decoder'
    decoder/layers.Dense.units = {self.x_dims}
    """

  @parameterized.named_parameters(
      ('kwarg_modules', True),
      ('dag_modules', False),
  )
  def test_build_layer(self, kwarg_modules):
    """Tests if layer builds properly and produces outputs of correct shape."""
    gin_config = (self.gin_config_kwarg_modules if kwarg_modules else
                  self.gin_config_dag_modules)
    with gin.unlock_config():
      gin.clear_config()
      gin.parse_config(gin_config)

    dag_layer = ConfigurableDAGLayer()
    outputs = dag_layer(self.inputs)
    self.assertIsInstance(outputs, dict)

    z = outputs['bottleneck']['z_bottleneck']
    x_rec = outputs['decoder']['reconstruction']
    x_rec2 = outputs['out']['reconstruction']

    # Confirm that layer generates correctly sized tensors.
    self.assertEqual(outputs['test_data'].shape, self.x.shape)
    self.assertEqual(outputs['inputs']['test_data'].shape, self.x.shape)
    self.assertEqual(x_rec.shape, self.x.shape)
    self.assertEqual(z.shape[-1], self.z_dims)
    self.assertAllClose(x_rec, x_rec2)

    # Confirm that variables are inherited by DAGLayer.
    self.assertLen(dag_layer.trainable_variables, 6)  # 3 weights, 3 biases.

if __name__ == '__main__':
  tf.test.main()
