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

"""Install ddsp."""

import os
import sys
import setuptools

# To enable importing version.py directly, we add its path to sys.path.
version_path = os.path.join(os.path.dirname(__file__), 'ddsp')
sys.path.append(version_path)
from version import __version__  # pylint: disable=g-import-not-at-top

setuptools.setup(
    name='ddsp',
    version=__version__,
    description='Differentiable Digital Signal Processing ',
    author='Google Inc.',
    author_email='no-reply@google.com',
    url='http://github.com/magenta/ddsp',
    license='Apache 2.0',
    packages=setuptools.find_packages(),
    package_data={
        '': ['*.gin'],
    },
    scripts=[],
    install_requires=[
        'absl-py',
        'cloudml-hypertune',
        'crepe>=0.0.11',
        'future',
        'gin-config>=0.3.0',
        'google-cloud-storage',
        'librosa',
        'pydub',
        'mir_eval',
        'note_seq',
        'numba < 0.50',  # temporary fix for librosa import
        'numpy',
        'scipy',
        'six',
        'tensorflow',
        'tensorflow-addons',
        'tensorflowjs',
        'tensorflow-probability',
        'tensorflow-datasets',
        'tflite_support'
    ],
    extras_require={
        'gcp': [
            'gevent', 'google-api-python-client', 'google-compute-engine',
            'oauth2client'
        ],
        'data_preparation': [
            'apache_beam',
            # TODO(jesseengel): Remove versioning when beam import is fixed.
            'pyparsing<=2.4.7'
        ],
        'test': ['pytest', 'pylint!=2.5.0'],
    },
    entry_points={
        'console_scripts': [
            'ddsp_export = ddsp.training.ddsp_export:console_entry_point',
            'ddsp_run = ddsp.training.ddsp_run:console_entry_point',
            'ddsp_prepare_tfrecord = ddsp.training.data_preparation.ddsp_prepare_tfrecord:console_entry_point',
            'ddsp_generate_synthetic_dataset = ddsp.training.data_preparation.ddsp_generate_synthetic_dataset:console_entry_point',
            'ddsp_ai_platform = ddsp.training.docker.ddsp_ai_platform:console_entry_point',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    tests_require=['pytest'],
    setup_requires=['pytest-runner'],
    keywords='audio dsp signalprocessing machinelearning music',
)
