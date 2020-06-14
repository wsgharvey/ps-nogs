import argparse

from utils import str2bool

parser = argparse.ArgumentParser(description='RAM')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    return arg

# special args not included in model name
special_arg = parser.add_argument_group('Special Params')
special_arg.add_argument('--mode', type=str, default='train',
                         choices=['train', 'test'],
                         help='Whether to train or test the model')
special_args = [a.dest for a in special_arg._actions]

# glimpse network params
nn_arg = add_argument_group('Glimpse Network Params')
nn_arg.add_argument('--glimpse_net_type', type=str,
                    help="Definition of glimpse network type.")
nn_arg.add_argument('--conv_depth', type=str, default=None,
                    help='Description of CNN (see nn.py).')
nn_arg.add_argument('--hidden_size', type=int,
                    help='RNN hidden layer size.')
nn_arg.add_argument('--classifier_size', type=str, default='',
                    help='Sequence of hidden sizes for final classifier layer. \
                    By default is a simple linear layer.')
nn_arg.add_argument('--glimpse_size', type=int, default=16,
                    help='size of extracted image patch')
nn_arg.add_argument('--num_classes', type=int, default=2,
                    help='# of epochs to train for')
nn_arg.add_argument('--num_glimpses', type=int, default=5,
                    help='# of glimpses')

# reinforce params
reinforce_arg = add_argument_group('Reinforce Params')
reinforce_arg.add_argument('--dist_type', type=str, default='categorical',
                           choices=['categorical', 'mixture'],
                           help='Type of distribution for location dist.')
reinforce_arg.add_argument('--std', type=float, default=0.00001,
                           help='gaussian policy standard deviation')
reinforce_arg.add_argument('--loc_mixture_components', type=int, default=30,
                           help='# of mixture components in location dist.')
reinforce_arg.add_argument('--entropy_regularizer', type=float, default=0,
                           help='Regularizer to control exploration/exploitation trade-off.')
reinforce_arg.add_argument('--M', type=float, default=10,
                           help='Monte Carlo sampling for valid and test sets')

# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--valid_size', type=str, default='default',
                      help='No. data points in valid set. Default value depends on dataset.')
data_arg.add_argument('--test_size', type=str, default='default',
                      help='No. data points in test set. Default value depends on dataset.')
data_arg.add_argument('--batch_size', type=int, default=64,
                      help='# of images in each batch of data')
data_arg.add_argument('--num_workers', type=int, default=7,
                      help='# of subprocesses to use for data loading')
data_arg.add_argument('--shuffle', type=str2bool, default=True,
                      help='Whether to shuffle the train and valid indices')

# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--is_feedforward', type=str2bool, default=False,
                       help='If true, trains/tests a feedforward network with a previously learned attention policy.')
train_arg.add_argument('--test_seqs', type=str2bool, default=False,
                       help='Test learned policy by running big dropout CNN on them.')
train_arg.add_argument('--feedforward_type', type=int, default=0, choices=[0, 1, 2, 3],
                       help='Feedforward with dropout (1), batchnorm (2), neither (0) or both (3).')
train_arg.add_argument('--attention_targets', type=str, default='exact',
                       help='Standard shit if exact. Otherwise, either pretrains RNN+CNN or uses pretrained one. See commented out choices.')
train_arg.add_argument('--attention_target_weight', type=float, default=1.0,
                       help='Weighting given to attention targets in loss.')
train_arg.add_argument('--momentum', type=float, default=0.5,
                       help='Nesterov momentum value')
train_arg.add_argument('--epochs', type=int, default=75,
                       help='# of epochs to train for')
train_arg.add_argument('--init_lr', type=float, default=3e-4,
                       help='Initial learning rate value')
train_arg.add_argument('--supervised_attention_prob', type=float, default=1.0,
                       help='Proportion of time to use targets for training the attention mechanism.')
train_arg.add_argument('--cuda', type=str2bool, default=False,
                       help='Use CUDA or not. Only works with CNN baselines.')

# other params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--best', type=str2bool, default=True,
                      help='Load best model or most recent for testing')
misc_arg.add_argument('--seed', type=int, default=1,
                      help='Seed to ensure reproducibility')
misc_arg.add_argument('--data_dir', type=str, default='../data',
                      help='Has been repurposed to load array of expected posterior entropies for training heuristic.')
misc_arg.add_argument('--dataset', type=str, default='celebhq',
                      help='Celeba or not', choices=['celeba', 'celebhq'])
misc_arg.add_argument('--ckpt_dir', type=str, default='ckpt',
                      help='Directory in which to save model checkpoints')
misc_arg.add_argument('--logs_dir', type=str, default='logs/',
                      help='Directory in which logs wil be stored')
misc_arg.add_argument('--resume', type=str2bool, default=False,
                      help='Whether to resume training from checkpoint')
misc_arg.add_argument('--celebhq_image_dir', type=str, default='../data/celebhq/images')
misc_arg.add_argument('--celebhq_image_type', type=str, default='png', choices=['png', 'npy', 'debug'])
misc_arg.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
misc_arg.add_argument('--hq_attr_path', type=str, default='../data/celebhq/annotations.p')
misc_arg.add_argument('--hq_sequences_dir', type=str, default='../optimal-sequences/celebhq/')
misc_arg.add_argument('--attr', type=str, help='selected attribute to predict')
misc_arg.add_argument('--image_size', type=int, default=224, help='image resolution')
misc_arg.add_argument('--grayscale', type=str2bool, default=False, help="Convert images to grayscale.")
misc_arg.add_argument('--n_optimal_seqs', type=int, default=None,
                      help="Number of optimal sequences to use (uses all available if not specified).")
misc_arg.add_argument('--fixed_attention_prop', type=float, default=None,
                      help="If not None, will fix all batches to contain this proportion of image with known optimal sequences.")
misc_arg.add_argument('--kde', type=str2bool, default=False,
                      help="If True, plot KDE instead of training/testing. Plotted on train or test dataset, as specified by --is_train")
misc_arg.add_argument('--pde', type=str2bool, default=False,
                      help="'Pixel Density Estimate'. Similar to KDE but plots images of how often each pixel is attended to.")
misc_arg.add_argument('--plot_sequences', type=str2bool, default=False,
                      help="Plot sequences of glimpses for first batch in either the training or test set.")
misc_arg.add_argument('--anneal_epochs', type=int, default=None,
                      help="Number of epochs to anneal supervision rate over. Will not anneal if not None. Must be used in conjunction with --fixed_attention_prop.")
misc_arg.add_argument('--test-tag', type=str, default=None, help="Tag to specify checkpoint for testing.")

default_valid_sizes = {'celebhq': 500}
default_test_sizes = {'celebhq': 2500}

def get_config():
    config = parser.parse_args()

    # set valid/test set sizes
    if config.valid_size == 'default':
        config.valid_size = default_valid_sizes[config.dataset]
    if config.test_size == 'default':
        config.test_size = default_test_sizes[config.dataset]
    config.valid_size = int(config.valid_size)
    config.test_size = int(config.test_size)
    config.classifier_size = [] if config.classifier_size == '' else \
        [int(l) for l in config.classifier_size.split(',')]
    return config

def get_config_fields():
    return sorted([action.dest for action in parser._actions if action.dest != 'help' and action.dest not in special_args])
