"""Write audio and spectrogram to tensorboard summary file."""
from ddsp.training.summaries import audio_summary, spectrogram_summary


def write_tensorboard_audio(writer, data, outputs, step, tag):
  """
  Write audio and spectrogram to tensorboard summary file.
  Args:
    writer: writer of the tensorboard summary file.
    data: the data batch from the dataset.
    outputs: outputs from the model output.
    step: current training step.
    tag: tag for the tensorboard summary. Usually is "train" or "test"

  """
  with writer.as_default():
    spectrogram_summary(data['audio'], outputs['synth_audio'], step, name='',
                        tag=f'{tag}/spectrogram_synth_coder')
    audio_summary(data['audio'], step, sample_rate=16000,
                  name=f'{tag}_audio_original')
    audio_summary(outputs['synth_audio'], step, sample_rate=16000,
                  name=f'{tag}_audio_synth_coder')
    if 'midi_audio' in outputs.keys():
      spectrogram_summary(data['audio'], outputs['midi_audio'], step, name='',
                          tag=f'{tag}/spectrogram_midi_decoder')
      audio_summary(outputs['midi_audio'], step, sample_rate=16000,
                    name=f'{tag}_audio_midi_decoder')
