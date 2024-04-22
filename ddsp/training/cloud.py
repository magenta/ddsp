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

"""Library of functions for training on Google Cloud AI-Platform."""

import os
import re

from absl import logging
from google.cloud import storage
import hypertune


def download_from_gstorage(gstorage_path, local_path):
  """Downloads a file from the bucket.

  Args:
    gstorage_path: Path to the file inside the bucket that needs to be
      downloaded. Format: gs://bucket-name/path/to/file.txt
    local_path: Local path where downloaded file should be stored.
  """
  gstorage_path = gstorage_path.strip('gs:/')
  bucket_name = gstorage_path.split('/')[0]
  blob_name = os.path.relpath(gstorage_path, bucket_name)

  storage_client = storage.Client()

  bucket = storage_client.bucket(bucket_name)
  blob = bucket.blob(blob_name)

  blob.download_to_filename(local_path)
  logging.info(
      'Downloaded file. Source: %s, Destination: %s',
      gstorage_path, local_path)


def make_file_paths_local(paths, local_directory):
  """Makes sure that given files are locally available.

  If a Cloud Storage path is provided, downloads the file and returns the new
  path relative to local_directory. If a local path is provided it is returns
  path with no modification.

  Args:
    paths: Single path or a list of paths.
    local_directory: Local path to the directory were downloaded files will be
      stored. Note that if you want to download gin configuration files

  Returns:
    Single local path or a list of local paths.
  """
  if isinstance(paths, str):
    if re.match('gs://*', paths):
      local_name = os.path.basename(paths)
      download_from_gstorage(paths, os.path.join(local_directory, local_name))
      return local_name
    else:
      return paths
  else:
    local_paths = []
    for path in paths:
      if re.match('gs://*', path):
        local_name = os.path.basename(path)
        download_from_gstorage(path, os.path.join(local_directory, local_name))
        local_paths.append(local_name)
      else:
        local_paths.append(path)
    return local_paths


def report_metric_to_hypertune(metric_value, step, tag='Loss'):
  """Use hypertune to report metrics for hyperparameter tuning."""
  hpt = hypertune.HyperTune()
  hpt.report_hyperparameter_tuning_metric(
      hyperparameter_metric_tag=tag,
      metric_value=metric_value,
      global_step=step)
