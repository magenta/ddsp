# Data Preparation

Scripts and libraries to prepare datasets for training. For some more example usage, check out the [train_autoencoder](../../colab/demos/train_autoencoder.ipynb) demo.


## Making a TFRecord dataset from your own sounds

For experiments from the original [ICLR 2020 paper](https://openreview.net/forum?id=B1x1ma4tDr), we need to do some preprocessing on the raw audio to get it into the correct format for training. This involves turning the full audio into short examples (4 seconds by default, but adjustable with flags), inferring the fundamental frequency (or \"pitch\") with [CREPE](http://github.com/marl/crepe), and computing the loudness. These features will then be stored in a sharded [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) file for easier loading. Depending on the amount of input audio, this process usually takes a few minutes.

```
ddsp_prepare_tfrecord \
--input_audio_filepatterns=/path/to/wavs/*wav,/path/to/mp3s/*mp3 \
--output_tfrecord_path=/path/to/output.tfrecord \
--num_shards=10 \
--alsologtostderr
```

## Making a TFRecord dataset from synthetic data

For experiments from the [ICML 2020 workshop paper](https://goo.gl/magenta/ddsp-inv) of performing pitch detection with inverse synthesis, we need to create a synthetic dataset to use for supervision. We must also specify the data generation function `generate_examples.generate_fn` we want to use as a `--gin_param` flag or in a `--gin_file`.

```
ddsp_generate_synthetic_dataset \
--output_tfrecord_path=/path/to/output.tfrecord \
--num_shards=1000 \
--gin_param="generate_examples.generate_fn = @generate_notes_v2" \
--num_examples=10000000 \
--alsologtostderr
```

