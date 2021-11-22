
from midi_ddsp.model import SynthCoder, MIDIAEInterpCond
from midi_ddsp.ddsp_inference import MelF0LDEncoder, F0LDEncoder, \
  FCHarmonicDecoder, FCStackHarmonicDecoder, Cnn8
import ddsp.training.nn as nn
from midi_ddsp.reverb_modules import ReverbModules
from midi_ddsp.decoder import MidiToHarmonicAutoregressiveDecoder, \
  MidiToF0AmpAutoregressiveDecoder, F0AmpAutoregressiveDecoder, \
  MidiNoiseToHarmonicDecoder
from midi_ddsp.midi_decoder import InterpretableConditioningMidiDecoder, \
  MidiDecoder
import ddsp.training
import tensorflow as tf
from data_handling.instrument_name_utils import NUM_INST


def get_midi_decoder(hp):
  z_preconditioning_stack = nn.FcStackOut(ch=512, layers=5, n_out=256)
  if hp.midi_decoder_decoder_net == 'dilated_conv':
    net = nn.DilatedConvStack(
      ch=256,
      layers_per_stack=5,
      stacks=4,
      norm_type='layer',
      conditional=True,
    )
    midi_decoder_decoder = ddsp.training.decoders.MidiToHarmonicDecoder(
      net=net,
      f0_residual=True,
      norm=True,
      output_splits=(('f0_midi', 1),
                     ('amplitudes', 1),
                     ('harmonic_distribution', 60),
                     ('noise_magnitudes', 65)),
    )
  elif hp.midi_decoder_decoder_net == 'noise_dilated_conv':
    net = nn.DilatedConvStack(
      ch=128,
      layers_per_stack=5,
      stacks=4,
      norm_type='layer',
      conditional=True,
    )
    midi_decoder_decoder = MidiNoiseToHarmonicDecoder(
      net=net,
      f0_residual=True,
      norm=True,
      output_splits=(('f0_midi', 1),
                     ('amplitudes', 1),
                     ('harmonic_distribution', 60),
                     ('noise_magnitudes', 65)),
    )
  elif hp.midi_decoder_decoder_net == 'rnn_synth_params':
    midi_decoder_decoder = MidiToHarmonicAutoregressiveDecoder(
      net_type=hp.params_generator_net_type)

  elif hp.midi_decoder_decoder_net == 'rnn_f0_ld':
    midi_decoder_decoder = MidiToF0AmpAutoregressiveDecoder()

  if hp.midi_decoder_type == 'interpretable_conditioning':
    midi_decoder = InterpretableConditioningMidiDecoder(
      decoder=midi_decoder_decoder,
      decoder_type=hp.midi_decoder_decoder_net,
      z_preconditioning_stack=z_preconditioning_stack,
      multi_instrument=hp.multi_instrument,
      position_code=hp.position_code,
      without_note_expression=hp.without_note_expression,
    )

  elif hp.midi_decoder_type == 'midi_decoder':
    midi_decoder_decoder = MidiToF0AmpAutoregressiveDecoder(
      sampling_method='random')
    midi_decoder = MidiDecoder(
      decoder=midi_decoder_decoder,
      multi_instrument=hp.multi_instrument,
    )
  elif hp.midi_decoder_type == 'unconditioned':
    midi_decoder_decoder = F0AmpAutoregressiveDecoder(sampling_method='random')
    midi_decoder = MidiDecoder(
      decoder=midi_decoder_decoder,
      multi_instrument=hp.multi_instrument,
    )

  return midi_decoder


def get_model(hp):
  if hp.midi_decoder_decoder_net == 'rnn_f0_ld' or \
        hp.midi_decoder_type == 'midi_decoder' or \
        hp.midi_decoder_type == 'unconditioned':
    use_f0_ld = True
    encoder = F0LDEncoder()
    decoder = FCStackHarmonicDecoder(hp.nhramonic, hp.nnoise)
  else:
    cnn = Cnn8(pool_size=(1, 2))
    encoder = MelF0LDEncoder(cnn, hp.nhid, hp.sample_rate, hp.win_length,
                             hp.hop_length, hp.n_fft, hp.num_mels,
                             hp.fmin)
    decoder = FCHarmonicDecoder(hp.nhramonic, hp.nnoise)
    use_f0_ld = False
  synth_coder = SynthCoder(encoder, decoder)

  midi_decoder = get_midi_decoder(hp)

  if hp.reverb:
    num_reverb = NUM_INST if hp.multi_instrument else 1
    reverb_module = ReverbModules(num_reverb=num_reverb,
                                  reverb_length=hp.reverb_length)
  else:
    reverb_module = None

  model = MIDIAEInterpCond(
    synth_coder=synth_coder,
    midi_decoder=midi_decoder,
    n_frames=hp.sequence_length,
    frame_size=hp.frame_size,
    sample_rate=hp.sample_rate,
    reverb_module=reverb_module,
    use_f0_ld=use_f0_ld
  )

  return model


def get_fake_data(hp):
  """Get the fake data for building the model."""
  fake_data = {
    'audio': tf.random.normal([1, hp.sequence_length * hp.frame_size]),
    'mel': tf.random.normal([1, hp.sequence_length, hp.num_mels]),
    'f0_hz': tf.random.normal([1, hp.sequence_length, 1]),
    'loudness_db': tf.random.normal([1, hp.sequence_length, 1]),
    'midi': tf.ones([1, hp.sequence_length], dtype=tf.int64),
    'instrument_id': tf.ones([1], dtype=tf.int64),
    'onsets': tf.zeros([1, hp.sequence_length], dtype=tf.int64),
    'offsets': tf.zeros([1, hp.sequence_length], dtype=tf.int64),
  }
  return fake_data, {'training': True}
