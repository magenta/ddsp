# Gin Configs: Datasets

This directory contains gin configs for datasets. Each file contains datasets to use for training, evaluation, and sampling.
Each training run with `ddsp_run.py` must be supplied both a model file and a dataset file, each with the `--gin_file` flag.
Evaluation and sampling jobs only require a dataset file.
