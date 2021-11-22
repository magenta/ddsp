"""Dump the dataset for training expression generator."""
import os
import argparse
from utils.training_utils import set_seed, get_hp
from utils.create_expression_generator_dataset_utils import dump_expression_generator_dataset
from hparams_synthesis_generator import hparams as hp
from midi_ddsp.get_model import get_model, get_fake_data

parser = argparse.ArgumentParser(description='Dump expression generator '
                                             'dataset.')
set_seed(1234)

if __name__ == '__main__':
    parser.add_argument('--model_path', type=str,
                        default=None,
                        help='The path to the model checkpoint.')
    parser.add_argument('--data_dir', type=str,
                        default=None,
                        help='The directory to the unbatched tfrecord dataset.')
    parser.add_argument('--output_dir', type=str,
                        default=None,
                        help='The output directory for dumping the expression '
                             'generator dataset.')
    # TODO: (yusongwu) add automatic note expression scaling
    args = parser.parse_args()
    model_path = args.model_path
    hp_dict = get_hp(os.path.join(os.path.dirname(model_path), 'train.log'))
    for k, v in hp_dict.items():
        setattr(hp, k, v)
    model = get_model(hp)
    _ = model._build(get_fake_data(hp))
    model.load_weights(model_path)

    print('Creating dataset for expression generator!')
    dump_expression_generator_dataset(model, data_dir=args.data_dir, output_dir=args.output_dir)
