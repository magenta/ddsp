# Copyright 2019 The DDSP Authors.
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
"""Base module for the differentiable digital signal processing library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Module imports.
from ddsp import core
from ddsp import effects
from ddsp import losses
from ddsp import processors
from ddsp import spectral_ops
from ddsp import synths

# Core imports.
from ddsp.core import fft_convolve
from ddsp.core import frequency_filter
from ddsp.core import frequency_impulse_response
from ddsp.core import harmonic_synthesis
from ddsp.core import linear_lookup
from ddsp.core import midi_to_hz
from ddsp.core import oscillator_bank
from ddsp.core import resample
from ddsp.core import sinc_filter
from ddsp.core import sinc_impulse_response
from ddsp.core import variable_length_delay
from ddsp.core import wavetable_synthesis

# Version number.
from ddsp.version import __version__
