# Demos

Here are colab notebooks for demonstrating neat things you can do with DDSP.

*   [timbre_transfer](https://colab.research.google.com/github/magenta/ddsp/blob/master/ddsp/colab/demos/timbre_transfer.ipynb):
    Convert audio between sound sources with pretrained models. Try turning your voice into a violin, or scratching your laptop and seeing how it sounds as a flute :). Pick from a selection of pretrained models or upload your own that you can train with the `train_autoencoder` demo.

*   [train_autoencoder](https://colab.research.google.com/github/magenta/ddsp/blob/master/ddsp/colab/demos/train_autoencoder.ipynb):
    Takes you through all the steps to convert audio files into a dataset and train your own DDSP autoencoder model. You can transfer data and models to/from google drive, and download a .zip file of your trained model to be used with the `timbre_transfer` demo.

*   [pitch_detection](https://colab.research.google.com/github/magenta/ddsp/blob/master/ddsp/colab/demos/pitch_detection.ipynb):
    Demonstration of self-supervised pitch detection models from [2020 ICML Workshop paper](https://openreview.net/forum?id=RlVTYWhsky7).
