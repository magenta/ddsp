#!/bin/bash
set -ex

orig_dir=$(pwd)
tmp_dir=$(mktemp -d -t ddsp-XXXX)
git clone https://github.com/magenta/ddsp.git $tmp_dir
cd $tmp_dir

python setup.py sdist
python setup.py bdist_wheel --universal
twine upload dist/*

cd $orig_dir
rm -rf $tmp_dir
