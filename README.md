<div align="center">
<img src="https://storage.googleapis.com/ddsp/github_images/ddsp_logo.png" width="200px" alt="logo"></img>
</div>

# DDSP: Differentiable Digital Signal Processing
[![Build Status](https://travis-ci.org/magenta/ddsp.svg?branch=master)](https://travis-ci.org/magenta/ddsp)

[**Demos**](./ddsp/colab/demos)
| [**Tutorials**](./ddsp/colab/tutorials)
| [**Blog Post**](https://magenta.tensorflow.org/ddsp)
| [**Overview**](#Overview)
| [**Installation**](#Installation)

DDSP is a library of differentiable versions of common DSP functions (such as
synthesizers, waveshapers, and filters). This allows these
interpretable elements to be used as part of an deep learning model, especially
as the output layers for audio generation.

## Getting Started

First, follow the steps in the [**Installation**](#Installation) section
to install the DDSP package and its dependencies. DDSP modules can be used to
generate and manipulate audio from neural network outputs as in this simple example:

```python
import ddsp

# Get synthesizer parameters from a neural network.
outputs = network(inputs)

# Initialize signal processors.
additive = ddsp.synths.Additive()

# Generates audio from additive synthesizer.
audio = additive(outputs['amplitudes'],
                 outputs['harmonic_distribution'],
                 outputs['f0_hz'])
```

### More resources

* [Check out the blog post 💻](https://magenta.tensorflow.org/ddsp)
* [Read the original paper 📄](https://arxiv.org/abs/2001.04643)
* [Listen to some examples 🔈](https://goo.gl/magenta/ddsp-examples)
* [Try out the timbre transfer demo 🎤->🎻](./ddsp/colab/demos/timbre_transfer.ipynb)


<a id='Tutorials'></a>
### Tutorials

The best place to start is the step-by-step tutorials for all the major library components that can be found in
[`ddsp/colab/tutorials`](./ddsp/colab/tutorials).

*   [0_processor](./ddsp/colab/tutorials/0_processor.ipynb):
    Introduction to the Processor class.
*   [1_synths_and_effects](./ddsp/colab/tutorials/1_synths_and_effects.ipynb):
    Example usage of processors.
*   [2_processor_group](./ddsp/colab/tutorials/2_processor_group.ipynb):
    Stringing processors together in a ProcessorGroup.
*   [3_training](./ddsp/colab/tutorials/3_training.ipynb):
    Example of training on a single sound.
*   [4_core_functions](./ddsp/colab/tutorials/4_core_functions.ipynb):
    Extensive examples for most of the core DDSP functions.


### Modules

The DDSP library consists of a [core library](./ddsp) (`ddsp/`) and a [self-contained training library](./ddsp/training) (`ddsp/training/`). The core library is split up into  into several modules:

*   [Core](./ddsp/core.py):
    All the differentiable DSP functions.
*   [Processors](./ddsp/processors.py):
    Base classes for Processor and ProcessorGroup.
*   [Synths](./ddsp/synths.py):
    Processors that generate audio from network outputs.
*   [Effects](./ddsp/effects.py):
    Processors that transform audio according to network outputs.
*   [Losses](./ddsp/losses.py):
    Loss functions relevant to DDSP applications.
*   [Spectral Ops](./ddsp/spectral_ops.py):
    Helper library of Fourier and related transforms.

Besides the tutorials, each module has its own test file that can be helpful for examples of usage.

<a id='Overview'></a>
# Overview

## Processor

The `Processor` is the main object type and preferred API of the DDSP library. It inherits from `tfkl.Layer` and can be used like any other differentiable module.

Unlike other layers, Processors (such as Synthesizers and Effects) specifically format their `inputs` into `controls` that are physically meaningful.
For instance, a synthesizer might need to remove frequencies above the [Nyquist frequency](https://en.wikipedia.org/wiki/Nyquist_frequency) to avoid [aliasing](https://en.wikipedia.org/wiki/Aliasing) or ensure that its amplitudes are strictly positive. To this end, they have the methods:

* `get_controls()`: inputs -> controls.
* `get_signal()`: controls -> signal.
* `__call__()`: inputs -> signal. (i.e. `get_signal(**get_controls())`)

Where:

* `inputs` is a variable number of tensor arguments (depending on processor). Often the outputs of a neural network.
* `controls` is a dictionary of tensors scaled and constrained specifically for the processor.
* `signal` is an output tensor (usually audio or control signal for another processor).

For example, here are of some inputs to an `Additive()` synthesizer:

<div align="center">
<img src="https://storage.googleapis.com/ddsp/github_images/example_inputs.png" width="800px" alt="logo"></img>
</div>

And here are the resulting controls after logarithmically scaling amplitudes, removing harmonics above the Nyquist frequency, and  normalizing the remaining harmonic distribution:

<div align="center">
<img src="https://storage.googleapis.com/ddsp/github_images/example_controls.png" width="800px" alt="logo"></img>
</div>

Notice that only 18 harmonics are nonzero (sample rate 16kHz, Nyquist 8kHz, 18\*440=7920Hz) and they sum to 1.0 at all times

## ProcessorGroup

Consider the situation where you want to string together a group of Processors.
Since Processors are just instances of `tfkl.Layer` you could use python control flow,
as you would with any other differentiable modules.

In the example below, we have an audio autoencoder that uses a
differentiable harmonic+noise synthesizer with reverb to generate audio for a multi-scale spectrogram reconstruction loss.

```python
import ddsp

# Get synthesizer parameters from the input audio.
outputs = network(audio_input)

# Initialize signal processors.
additive = ddsp.synths.Additive()
filtered_noise = ddsp.synths.FilteredNoise()
reverb = ddsp.effects.TrainableReverb()
spectral_loss = ddsp.losses.SpectralLoss()

# Generate audio.
audio_additive = additive(outputs['amplitudes'],
                          outputs['harmonic_distribution'],
                          outputs['f0_hz'])
audio_noise = filtered_noise(outputs['magnitudes'])
audio = audio_additive + audio_noise
audio = reverb(audio)

# Multi-scale spectrogram reconstruction loss.
loss = spectral_loss(audio, audio_input)
```

### ProcessorGroup (with a list)

A `ProcessorGroup` allows specifies a as a Directed Acyclic Graph (DAG) of processors. The main advantage of using a ProcessorGroup is that the entire signal processing chain can be specified in a `.gin` file, removing the need to write code in python for every different configuration of processors.


You can specify the DAG as a list of tuples `dag = [(processor, ['input1', 'input2', ...]), ...]` where `processor` is an Processor instance, and `['input1', 'input2', ...]` is a list of strings specifying input arguments. The output signal of each processor can be referenced as an input by the string `'processor_name/signal'` where processor_name is the name of the processor at construction. The ProcessorGroup takes a dictionary of inputs, who keys can be referenced in the DAG.



```python
import ddsp
import gin

# Get synthesizer parameters from the input audio.
outputs = network(audio_input)

# Initialize signal processors.
additive = ddsp.synths.Additive()
filtered_noise = ddsp.synths.FilteredNoise()
add = ddsp.processors.Add()
reverb = ddsp.effects.TrainableReverb()
spectral_loss = ddsp.losses.SpectralLoss()

# Processor group DAG
dag = [
  (additive,
   ['amps', 'harmonic_distribution', 'f0_hz']),
  (filtered_noise,
   ['magnitudes']),
  (add,
   ['additive/signal', 'filtered_noise/signal']),
  (reverb,
   ['add/signal'])
]
processor_group = ddsp.processors.ProcessorGroup(dag=dag)

# Generate audio.
audio = processor_group(outputs)

# Multi-scale spectrogram reconstruction loss.
loss = spectral_loss(audio, audio_input)
```


### ProcessorGroup (with `gin`)

The main advantage of a ProcessorGroup is that it can be defined with a `.gin` file, allowing flexible configurations without having to write new python code for every new DAG.

In the example below we pretend we have an external file written, which we treat here as a string. Now, after parsing the gin file, the ProcessorGroup will have its arguments configured on construction.

```python
import ddsp
import gin

gin_config = """
import ddsp

processors.ProcessorGroup.dag = [
  (@ddsp.synths.Additive(),
   ['amplitudes', 'harmonic_distribution', 'f0_hz']),
  (@ddsp.synths.FilteredNoise(),
   ['magnitudes']),
  (@ddsp.processors.Add(),
   ['filtered_noise/signal', 'additive/signal']),
  (@ddsp.effects.TrainableReverb(),
   ['add/signal'])
]
"""

with gin.unlock_config():
  gin.parse_config(gin_config)

# Get synthesizer parameters from the input audio.
outputs = network(audio_input)

# Initialize signal processors, arguments are configured by gin.
processor_group = ddsp.processors.ProcessorGroup()

# Generate audio.
audio = processor_group(outputs)

# Multi-scale spectrogram reconstruction loss.
loss = spectral_loss(audio, audio_input)
```

## A word about `gin`...

The [gin](https://github.com/google/gin-config) library is a "super power" of
dependency injection, and we find it very helpful for our experiments, but
with great power comes great responsibility. There are two methods for injecting dependencies with gin.

* `@gin.configurable`
makes a function globally configurable, such that *anywhere* the function or
object is called, gin sets its default arguments/constructor values. This can
lead to a lot of unintended side-effects.

* `@gin.register` registers a function
or object with gin, and only sets the default argument values when the function or object itself is used as an argument to another function.

To "use gin responsibly", by wrapping most
functions with `@gin.register` so that they can be specified as arguments of more "global" `@gin.configurable` functions/objects such as `ProcessorGroup` in the main library and
`Model`, `train()`, `evaluate()`, and `sample()` in [`ddsp/training`](./ddsp/training).

As you can see in the code, this allows us to flexibly define hyperparameters of
most functions without worrying about side-effects. One exception is `ddsp.core.cumsum` where we configure special optimizations for TPU.

<a id='Installation'></a>
# Installation
Requires tensorflow version >= 2.1.0, but the core library runs in either eager or graph mode.

```bash
sudo apt-get install libsndfile-dev
pip install --upgrade pip
pip install --upgrade ddsp
```

<a id='Contributing'></a>
# Contributing

We're eager to collaborate with you! See [`CONTRIBUTING.md`](CONTRIBUTING.md)
for a guide on how to contribute.

<a id='Citation'></a>
# Citation

If you use this code please cite it as:

```latex
@inproceedings{
  engel2020ddsp,
  title={DDSP: Differentiable Digital Signal Processing},
  author={Jesse Engel and Lamtharn (Hanoi) Hantrakul and Chenjie Gu and Adam Roberts},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=B1x1ma4tDr}
}
```

<a id='Disclaimer'></a>
# Disclaimer

This is not an official Google product.
