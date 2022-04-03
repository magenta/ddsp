# DDSP: Differentiable Digital Signal Processing
_ICLR 2020, ([paper](g.co/magenta/ddsp))_

Gin configs for reproducing the results of the paper.

Each config file specifies both the dataset and the model, so only a single --gin_file flag is required.

## Configs

* `nsynth_ae.gin`: Autoencoder (with latent z, and f0 given by CREPE) on NSynth dataset.
* `nsynth_ae_abs.gin`: Deprecated. Please install v0.0.6 to use this config. Autoencoder (with latent z, and f0 inferred by model) on NSynth dataset. For improved version, see [icml2020](./../icml2020/) paper.
* `solo_instrument.gin`: Decoder (with no z, and f0 given by CREPE) on your own dataset of a monophonic instrument. Make dataset with `ddsp/training/data_preparation/ddsp_prepare_tfrecord.py`.
* `tiny_instrument.gin`: Same as `solo_instrument.gin`, but with much smaller model.


