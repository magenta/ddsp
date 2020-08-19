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
"""Tests for ddsp.training.test_util."""

import json
import os
import tensorflow.compat.v2 as tf
from tensorflow.python.eager import context
import train_util


class GetStrategyTest(tf.test.TestCase):

  def test_multiworker_strategy_is_set(self):
    """Check if proper strategy is returned and cluster config is set."""
    # Context reset needed after last test
    context._reset_context() # pylint: disable=protected-access

    strategy = train_util.get_strategy(
        cluster_config=(
            '{"cluster": {"worker": '
            '["worker0.example.com:2221", "worker1.example.com:2222"]},'
            '"task": {"type": "worker", "index": 0}}'))
    self.assertIsInstance(
        strategy,
        tf.distribute.experimental.MultiWorkerMirroredStrategy)
    self.assertDictEqual(
        strategy.cluster_resolver.cluster_spec().as_dict(),
        {"worker": ["worker0.example.com:2221", "worker1.example.com:2222"]})
    self.assertEqual(strategy.cluster_resolver.task_type, "worker")
    self.assertEqual(strategy.cluster_resolver.task_id, 0)

  def test_tf_config_is_overwritten(self):
    """Check if cluster config from TF_CONFIG is overwritten."""
    # Context reset needed after last test
    context._reset_context() # pylint: disable=protected-access

    os.environ["TF_CONFIG"] = json.dumps(
        {"cluster": {"worker": ["worker0.example.com:2221",
                                "worker1.example.com:2222"]},
         "task": {"type": "worker", "index": 0}})
    strategy = train_util.get_strategy(
        cluster_config=(
            '{"cluster": {"worker": ["worker0.example.com:2221"], '
            '"chief": ["chief.example.com:2222"]}, '
            '"task": {"type": "chief", "index": 0}}'))
    self.assertDictEqual(
        strategy.cluster_resolver.cluster_spec().as_dict(),
        {"worker": ["worker0.example.com:2221"],
         "chief": ["chief.example.com:2222"]})
    self.assertEqual(strategy.cluster_resolver.task_type, "chief")
    self.assertEqual(strategy.cluster_resolver.task_id, 0)

  def test_defaulting_to_mirrored_strategy(self):
    """Check that single worker strategy is returned if no args provided."""
    # Context reset needed after last test
    context._reset_context() # pylint: disable=protected-access

    strategy = train_util.get_strategy()
    self.assertIsInstance(strategy, tf.distribute.MirroredStrategy)

if __name__ == "__main__":
  tf.test.main()
