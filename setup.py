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
"""Install ddsp."""

import setuptools

setuptools.setup(
    name='ddsp',
    version='0.0.0',
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
        'crepe',
        'future',
        'gin-config>=0.3.0',
        'librosa',
        'pydub',
        'numpy',
        'scipy',
        'six',
        'tensorflow>=2.1.0',
        # TODO(adarob): Switch to tensorflow_datasets once includes nsynth 2.3.
        'tfds-nightly',
    ],
    extras_require={
        'gcp': ['gevent', 'google-api-python-client', 'google-compute-engine',
                'google-cloud-storage', 'oauth2client'],
        'data_preparation': ['apache_beam'],
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'ddsp_run = ddsp.training.ddsp_run:console_entry_point',
            'ddsp_prepare_tfrecord = ddsp.training.data_preparation.prepare_tfrecord:console_entry_point',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    # TODO(adarob): Add pylint.
    tests_require=['pytest'],
    setup_requires=['pytest-runner'],
    keywords='audio dsp signalprocessing machinelearning music',
)
