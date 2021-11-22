"""Utility functions for file io and file path reading."""

import os
import shutil
import pickle
import json


def get_folder_name(path, num=1):
  """
  Get the name of the folder n levels above the given path.
  Example: a/b/c/d.txt, num=1 -> c, num=2 -> b, ...
  Args:
    path: a file path.
    num: the number of upper directories.

  Returns: the folder name for that level.

  """
  for i in range(num):
    path = os.path.dirname(path)
  return os.path.basename(path)


def copy_file_to_folder(file_path, dst_dir):
  save_path = os.path.join(dst_dir, os.path.basename(file_path))
  shutil.copy(file_path, save_path)


def pickle_dump(obj, path):
  file = open(path, 'wb')
  pickle.dump(obj, file)
  file.close()


def pickle_load(path):
  file = open(path, 'rb')
  data = pickle.load(file)
  file.close()
  return data


def json_dump(data_json, json_save_path):
  with open(json_save_path, 'w') as f:
    f.write(data_json)
    f.close()


def json_load(json_path):
  with open(json_path, 'r') as f:
    data = json.load(f)
    f.close()
  return data


def write_str_lines(save_path, lines):
  lines = [l + '\n' for l in lines]
  with open(save_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)
