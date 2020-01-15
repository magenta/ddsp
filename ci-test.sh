#!/bin/bash

# Fail on any error.
set -e

pip install -e .[data_preparation,test]
pytest
pylint ddsp
