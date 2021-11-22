"""The hyperparameters for training synthesis generator.
For details of each hyperparameters, please see the argpars """

class hparams:
  # Learning parameters
  batch_size = 4
  clip_grad = 2.5
  lr = 3e-4
  seed = 1111

  # Training parameters
  training_steps = 100000
  log_interval = 100
  checkpoint_save_interval = 5000
  eval_interval = 5000
  mode = 'train'  # train, eval
  data_dir = None
  restore_path = None

  # Model Parameters
  nhid = 256
  sequence_length = 1000
  train_synth_coder_first = True
  midi_audio_loss = True
  add_synth_loss = False
  synth_params_loss = False
  midi_decoder_type = 'interpretable_conditioning'
  params_encoder_note_pooling = True
  params_encoder_type = 'conv'
  params_generator_net_type = 'conv'
  position_code = 'index_length'
  midi_decoder_decoder_net = 'rnn_synth_params'
  multi_instrument = True
  instrument = 'vn'
  synth_coder_training_steps = 300
  lambda_recon = 1.0
  use_gan = True
  lambda_G = 1
  sg_z = True
  lr_disc = 1e-4
  write_tfrecord_audio = False
  without_note_expression = False
  discriminator_dim = 256

  # Synthesis & DSP parameters
  nhramonic = 60
  nnoise = 65
  reverb = True
  reverb_length = 48000
  num_mels = 64
  n_fft = 1024
  sample_rate = 16000
  frame_size = 64
  hop_length = frame_size
  win_length = hop_length * 2
  frame_shift_ms = 1000 / sample_rate * frame_size
  fmin = 40
