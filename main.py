from test import *
from train import *
import constants as keys
import argparse
import random

# Setting all the seeds
my_seed = 0
np.random.seed(my_seed)
random.seed(my_seed)
torch.manual_seed(my_seed)
torch.cuda.manual_seed(my_seed)
torch.backends.cudnn.benchmark = False

# Parsing chosen dataset and train modality
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', required=True, help='Name of the selected UCI dataset')
parser.add_argument('--mode', required=True, help='Train modality')
args = parser.parse_args()
dataset = args.dataset
training_mode = True if args.mode == 'train' else False

# Obtaining the best hyperparameters for the chosen dataset
lr, bs, epochs = keys.HYPERPARAMETER_CROSS_DICT[dataset]

# If we are in training mode we add the dropout 0.0, since we want to train a model without dropout (dr=0.0)
drs = keys.DROPOUT_RATES
if training_mode:
    drs += [0.0]

# Iterating over the dropout rates
for dr in drs:
    print('Starting with dataset {}'.format(dataset))
    if training_mode:
        train(dataset, dr, lr, bs, epochs, activate_pipeline=True)      # Train a model with dr=dr
    else:
        test(dataset, dr, lr, bs, mc_size=1000, dr_test=dr)     # Test embedded (dr_training=dr)
        test(dataset, 0.0, lr, bs, mc_size=1000, dr_test=dr)    # Test injected (dr_training=0)

