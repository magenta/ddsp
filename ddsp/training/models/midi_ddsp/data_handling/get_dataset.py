"""Create dataset and dataloader for the URMP dataset to train
the synthesis generator."""

from ddsp.training.data import UrmpMidi
import math


def get_tfrecord_length(dataloader):
  """Return the length of a dataloader."""
  train_dataset_single = dataloader.get_batch(batch_size=1,
                                              shuffle=False,
                                              repeats=1)
  c = 0
  for _ in train_dataset_single:
    c += 1
  return c


def get_dataset(hp):
  """Return dataloader based on hyperparameters."""
  data_dir = hp.data_dir
  train_data_loader = UrmpMidi(data_dir, instrument_key=hp.instrument,
                               split='train', suffix='batched')
  training_data = train_data_loader.get_batch(batch_size=hp.batch_size,
                                              shuffle=True,
                                              repeats=-1,
                                              drop_remainder=True)
  test_data_loader = UrmpMidi(data_dir, instrument_key=hp.instrument,
                              split='test', suffix='batched')
  evaluation_data = test_data_loader.get_batch(batch_size=hp.batch_size * 3,
                                               shuffle=True,
                                               repeats=1,
                                               drop_remainder=False)

  length_training_data = get_tfrecord_length(train_data_loader)
  length_evaluation_data = get_tfrecord_length(test_data_loader)
  length_training_data = math.ceil(length_training_data / hp.batch_size)
  length_evaluation_data = math.ceil(
    length_evaluation_data / (hp.batch_size * 3))

  training_data = iter(training_data)

  return training_data, length_training_data, evaluation_data, \
         length_evaluation_data
