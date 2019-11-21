# Realtime NSynth (rt_nsynth)

This directory contains the code for training models to synthesize audio from
realtime control features (fundamental frequency, loudness, latents), for
monophonic audio signals.

## Gin: Setting hyperparameters

Hyperparameters for training runs are specified through `.gin` files in the
`gin` directory. You can pass multiple of these to the training binary with the
`--gin_file` comandline flag. Paths are relative to the `gin` directory, for
example `--gin_file=additive/base.gin` You can also modify hyperparameters via
the `--gin_param` commandline flag.

When training, all gin parameters in the
[operative configuration](https://github.com/google/gin-config/blob/master/docs/index.md#retrieving-operative-parameter-values)
will be saved to the `${MODEL_DIR}/operative_config-0.gin` file. See
[this doc](https://github.com/google/gin-config/blob/master/docs/index.md#saving-gins-operative-config-to-a-file-and-tensorboard)
for more details.
