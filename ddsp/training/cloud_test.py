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

"""Tests for ddsp.training.cloud."""

from unittest import mock

from ddsp.training import cloud
import tensorflow.compat.v2 as tf


class MakeFilePathsLocalTest(tf.test.TestCase):

  @mock.patch.object(cloud, 'download_from_gstorage', autospec=True)
  def test_single_path_handling(self, download_from_gstorage_function):
    """Tests that function returns a single value if given single value."""
    path = cloud.make_file_paths_local(
        'gs://bucket-name/bucket/dir/some_file.gin',
        'gin/search/path')
    download_from_gstorage_function.assert_called_once()
    self.assertEqual(path, 'some_file.gin')

  @mock.patch.object(cloud, 'download_from_gstorage', autospec=True)
  def test_single_local_path_handling(self, download_from_gstorage_function):
    """Tests that function does nothing if given local file path."""
    path = cloud.make_file_paths_local(
        'local_file.gin',
        'gin/search/path')
    download_from_gstorage_function.assert_not_called()
    self.assertEqual(path, 'local_file.gin')

  @mock.patch.object(cloud, 'download_from_gstorage', autospec=True)
  def test_single_path_in_list_handling(self, download_from_gstorage_function):
    """Tests that function returns a single-element list if given one."""
    path = cloud.make_file_paths_local(
        ['gs://bucket-name/bucket/dir/some_file.gin'],
        'gin/search/path')
    download_from_gstorage_function.assert_called_once()
    self.assertNotIsInstance(path, str)
    self.assertListEqual(path, ['some_file.gin'])

  @mock.patch.object(cloud, 'download_from_gstorage', autospec=True)
  def test_more_paths_in_list_handling(self, download_from_gstorage_function):
    """Tests that function handle both local and gstorage paths in one list."""
    paths = cloud.make_file_paths_local(
        ['gs://bucket-name/bucket/dir/first_file.gin',
         'local_file.gin',
         'gs://bucket-name/bucket/dir/second_file.gin'],
        'gin/search/path')
    self.assertEqual(download_from_gstorage_function.call_count, 2)
    download_from_gstorage_function.assert_has_calls(
        [mock.call('gs://bucket-name/bucket/dir/first_file.gin', mock.ANY),
         mock.call('gs://bucket-name/bucket/dir/second_file.gin', mock.ANY)])
    self.assertListEqual(
        paths,
        ['first_file.gin', 'local_file.gin', 'second_file.gin'])


if __name__ == '__main__':
  tf.test.main()
