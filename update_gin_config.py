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

"""Script to update old '.gin' config files to work with current codebase.

When there are changes in a function signature (adding or removing kwargs),
those changes need to be reflected in gin config files. There is no way to
automatically do this from within gin.

This script maintains some backwards compatability, by keeping track of changes
to function signatures and automatically updating gin config files to
incorporate those changes.

In particular, this is useful for operative_config-*.gin files that are created
during training, to allow loading of old models.

This script will update any target gin config files, and leave copies of the old
files for safe-keeping.

Usage:
======
python update_gin_config.py path/to/config/operative_config-*.gin
"""

import os
import re

from absl import app
from absl import flags
from tensorflow import io

flags.DEFINE_string('path', '/tmp/operative_config-*.gin', 'Path to gin '
                    'config files. May be a path to a file, a or a glob '
                    'expression.')
flags.DEFINE_boolean('overwrite', False, 'Overwrite orginal files.')


FLAGS = flags.FLAGS
gfile = io.gfile


# ==============================================================================
# Updates to perform
# ==============================================================================
# Remove lines with any of these strings in them.
REMOVE = [
    'SpectralLoss.delta_delta_freq_weight',
    'SpectralLoss.delta_delta_time_weight',
    'DilatedConvEncoder.resample',
    'DilatedConvDecoder.resample',
]


# Perform these line-by-line substitutions, (Regex, Replacement String).
SUBSTITUTE = [
    ('ZRnnFcDecoder', 'RnnFcDecoder'),
]


# Add the following kwargs, (GinConfigurable, kwarg, value).
# -> 'GinConfigurable.kwarg = value'
ADD = [
    ('RnnFcDecoder', 'input_keys', '("f0_scaled", "ld_scaled")'),
]


# ==============================================================================
# Main Program
# ==============================================================================
def add_kwarg(lines, gin_configurable, kwarg, value):
  """Return the line where a GinConfigurable first appears."""
  gin_kwarg = gin_configurable + '.' + kwarg
  new_line = gin_kwarg + ' = ' + value + '\n'
  configurable_present = any([gin_configurable in line for line in lines])
  kwarg_present = any([gin_kwarg in line for line in lines])
  if configurable_present and not kwarg_present:
    # Add to the bottom of the config.
    lines.append('\n' + new_line)
    print(f'Added: {new_line.rstrip()}')
  elif configurable_present and kwarg_present:
    print(f'Skipped Add: {new_line.rstrip()}, {gin_kwarg} already present.')
  else:
    print(f'Skipped Add: {new_line.rstrip()}, {gin_configurable} not present.')


def main(argv):
  # Parse input args.
  if len(argv) > 2:
    raise app.UsageError('Too many command-line arguments.')
  elif len(argv) == 2:
    path = argv[1]
  else:
    path = FLAGS.path

  # Get a list of files that match the pattern.
  files = gfile.glob(path)

  for fpath in files:
    # Create a new file path.
    dirname, filename = os.path.split(fpath)
    new_filename = filename if FLAGS.overwrite else 'updated_' + filename
    new_fpath = os.path.join(dirname, new_filename)
    print(f'\nUpdating: \n{fpath} -> \n{new_fpath}')
    print('================')

    # Read old config.
    with gfile.GFile(fpath, 'r') as f:
      lines = f.readlines()

    # Make new config.
    new_lines = []
    for line in lines:
      # Remove lines with old arguments.
      if any([tag in line for tag in REMOVE]):
        print(f'Removed: {line.rstrip()}')
        continue

      # Substitute.
      for regex, sub in SUBSTITUTE:
        old_line = line
        line, n = re.subn(regex, sub, line)
        if n:
          print(f'Swapped: {old_line.rstrip()} -> {line.rstrip()}')

      # Append the new line.
      new_lines.append(line)

    # Add new lines after substitutions.
    for gin_configurable, kwarg, value in ADD:
      add_kwarg(new_lines, gin_configurable, kwarg, value)

    # Delete target file if it exists.
    if gfile.exists(new_fpath):
      gfile.remove(new_fpath)

    # Write to a new file.
    with gfile.GFile(new_fpath, 'w') as f:
      _ = f.write(''.join(new_lines))


if __name__ == '__main__':
  app.run(main)
