# DDSP: Differentiable Digital Signal Processing

DDSP is a library of differentiable versions of common DSP functions (such as
synthesizers, waveshapers, and filters). This allows these
interpretable elements to be used as part of an deep learning model, especially
as the output layers for audio generation.


## Example
In the example below, we construct an audio autoencoder that uses a
differentiable harmonic synthesizer and multi-scale spectrogram reconstrurction
loss.

```python
import ddsp

# Initialize signal processors.
additive = ddsp.synths.Additve()
reverb = ddsp.effects.TrainableReverb()
loss_fn = ddsp.losses.SpectralLoss()

# Get synthesizer parameters from the input audio.
outputs = network(audio_input)

# Generate audio.
audio = additive(outputs['amplitudes'],
                 outputs['harmonic_distribution'],
                 outputs['f0_hz'])
audio = reverb(audio)

# Multi-scale spectrogram reconstruction loss.
loss = loss_fn(audio, audio_input)
```

## Structure of the DDSP Library

DDSP is composed of several modules:

*   [Core](https://github.com/):
    All the core differentiable DSP functions.
*   [Processors](https://github.com/):
    Base classes for Processor and ProcessorGroup.
*   [Synths](https://github.com/):
    Processors that generate audio from network outputs.
*   [Effects](https://github.com/):
    Processors that transorm audio according to network outputs.
*   [Losses](https://github.com/):
    Loss functions relevant to DDSP applications.
*   [Spectral_Ops](https://github.com/):
    Helper library of Fourier and related transforms.
*   [pretrained_models](https://github.com/):
    Helper library of models for perceptual loss functions.


<a id='Installation'></a>
## Installation
`pip install ddsp`


<a id='Contributing'></a>
## Contributing

We're eager to collaborate with you! See [`CONTRIBUTING.md`](CONTRIBUTING.md)
for a guide on how to contribute.

<a id='Citation'></a>
## Citation

If you use this code please cite it as:

```
@misc{DDSP,
  title = {{DDSP}: Differentiable Digital Signal Processing},
  author = "{Jesse Engel, Lamtharn Hantrakul, Chenjie Gu, & Adam Roberts}",
  howpublished = {\url{}},
  url = "https://github.com/magenta/ddsp",
  year = 2019,
}
```

<a id='Disclaimer'></a>
## Disclaimer

This is not an official Google product.
