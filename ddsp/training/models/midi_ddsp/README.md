<div align="center">
<img src="https://midi-ddsp.github.io/pics/midi-ddsp-logo.png" width="200px" alt="logo"></img>
</div>

# MIDI-DDSP: Detailed Control of Musical Performance via Hierarchical Modeling

This directory contains the code for [paper](https://openreview.net/pdf?id=UseMOjWENv) "MIDI-DDSP: Detailed Control of
Musical Performance via Hierarchical Modeling" submitted to ICLR 2022.

## Train MIDI-DDSP

To train MIDI-DDSP, please first clone the ddsp repository:

```
git clone https://github.com/magenta/ddsp.git
```

Then, enter this directory and install the libraries needed:

```
pip install -r requirements.txt
```

or

```
pip install ddsp pretty_midi music21 librosa pandas
```

Please update to the latest version of DDSP (`pip install --upgrade ddsp`) if there is an existing version. If you
failed to install ddsp, please check [here](https://github.com/magenta/ddsp/blob/main/README.md#installation)
for more information. Also, please remember to [install CUDA and cuDNN](https://www.tensorflow.org/install/gpu) if you
are using GPU.

With environment installed, please download the tfrecord files for the URMP dataset in
[here](https://console.cloud.google.com/storage/browser/magentadata/datasets/urmp/urmp_20210324) to the `data` folder in
your cloned repository using the following commands:

```
gsutil cp gs://magentadata/datasets/urmp/urmp_20210324/* ./data
```

Please check [here](https://cloud.google.com/storage/docs/gsutil) for how to install and use `gsutil`.

Finally, you can run the script `train_midi_ddsp.sh` to train the exact same model we used in the paper:

```
sh ./train_midi_ddsp.sh
```

Side note:

We recommend using a GPU with more than 24G of memory when training synthesis generator in batch size of 16. For a GPU
with less memory, please consider using a smaller batch size and change the batch size in `train_midi_ddsp.sh`.

If one download the dataset to a different location, please change the `data_dir` parameter in `train_midi_ddsp.sh`.

The training of MIDI-DDSP takes approximately 24 hours on a single RTX 8000. The training code for now does not support
multi-GPU training.

## Try to play with MIDI-DDSP yourself!

Please try out MIDI-DDSP in Colab notebooks!

MIDI synthesis and adjustments:

TODO (cannot opensource the link now because it contains my GitHub token)

Bach chorales quartet synthesis:

TODO

## Command-line MIDI synthesis

On can use the MIDI-DDSP as a command-line MIDI synthesizer just like FluidSynth. Please first download the zip file
containing the model checkpoints
in [here](https://drive.google.com/file/d/1HbS2fQItqIeeTqalVd65qvw8PeuvYSYz/view?usp=sharing), unzip and put in some
path which we will refer to `<path-to-checkpoint-folder>`.

To use command-line synthesis to synthesize a midi file, run the following command:

```
synthesize_midi.py \
--midi_path <path-to-midi> \
--output_dir <output-dir> \
--synthesis_generator_weight_path <path-to-checkpoint-folder/50000>
--expression_generator_weight_path <path-to-checkpoint-folder/5000>
--use_fluidsynth
```

The command line also enables synthesize a folder of midi files. For more options, please
see `synthesize_midi.py --help`.

TODO: detailed instructions for using command line

[comment]: <> (## TODO: 0. Add script, dealing with model weight download, 1. Change the training loop, 2. Support multi-gpu training)