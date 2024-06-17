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

"""Base module for the differentiable digital signal processing library."""

# Module imports.
from ddsp import core
from ddsp import dags
from ddsp import effects
from ddsp import losses
from ddsp import processors
from ddsp import spectral_ops
from ddsp import synths

# Version number.
from ddsp.version import __version__
