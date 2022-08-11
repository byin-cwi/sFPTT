
import os
import sys
import argparse
import math
import shutil
import time
import logging
from io import open

import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim


from utils import get_xt
from add_task_nn import *
# from snn_models_LIF5 import *
from datasets import data_generator, adding_problem_generator
from train_addTask import *
from torch.optim import lr_scheduler
from train_addTask import *


def ptb_count_parameters(model):
    return sum(p.numel() for p in model.rnn.parameters() if p.requires_grad)

parser = argparse.ArgumentParser(description='Sequential Decision Making..')

parser.add_argument('--alpha', type=float, default=.5, help='Alpha')
parser.add_argument('--beta', type=float, default=0.5, help='Beta')
parser.add_argument('--rho', type=float, default=0.0, help='Rho')
parser.add_argument('--lmbda', type=float, default=2.0, help='Lambda')
parser.add_argument('--debias', action='store_true', help='FedDyn debias algorithm')
parser.add_argument('--K', type=int, default=1, help='Number of iterations for debias algorithm')

parser.add_argument('--nlayers', type=int, default=1, #2,
                    help='number of layers')
parser.add_argument('--bptt', type=int, default=300, #35,
                    help='sequence length')

parser.add_argument('--nhid', type=int, default=128,
                    help='number of hidden units per layer')

parser.add_argument('--lr', type=float, default=3e-3,
                    help='initial learning rate (default: 4e-3)')
parser.add_argument('--clip', type=float, default=1., #0.5,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit (default: 200)')
parser.add_argument('--parts', type=int, default=10,
                    help='Parts to split the sequential input into (default: 10)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')


parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                    help='batch size')

parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')


parser.add_argument('--wnorm', action='store_false',
                    help='use weight normalization (default: True)')
parser.add_argument('--temporalwdrop', action='store_false',
                    help='only drop the temporal weights (default: True)')
parser.add_argument('--wdecay', type=float, default=0.0,
                    help='weight decay')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use')
parser.add_argument('--when', nargs='+', type=int, default=[10,20,40,70,90,120,150,170],#[30,70,120],#[10,20,50, 75, 90],
                    help='When to decay the learning rate')

parser.add_argument('--load', type=str, default='',
                    help='path to load the model')
parser.add_argument('--save', type=str, default='./models/',
                    help='path to load the model')

parser.add_argument('--per_ex_stats', action='store_true',
                    help='Use per example stats to compute the KL loss (default: False)')
parser.add_argument('--permute', action='store_true',
                    help='use permuted dataset (default: False)')
parser.add_argument('--dataset', type=str, default='MNIST-10',
                    help='dataset to use')
parser.add_argument('--dataroot', type=str, 
                    default='./data/',
                    help='root location of the dataset')
args = parser.parse_args()

args.cuda = True

exp_name = args.dataset + '-nhid-' + str(args.nhid) + '-parts-' + str(args.parts) + '-optim-' + args.optim
exp_name += '-B-' + str(args.batch_size) + '-E-' + str(args.epochs) + '-K-' + str(args.K)
exp_name += '-alpha-' + str(args.alpha) + '-beta-' + str(args.beta)
# exp_name += 
if args.permute:
    exp_name += '-perm-' + str(args.permute)
if args.per_ex_stats:
    exp_name += '-per-ex-stats-'
if args.debias:
    exp_name += '-debias-'

prefix = args.save + exp_name

logger = logging.getLogger('trainer')

file_log_handler = logging.FileHandler( './logs/logfile-' + exp_name + '.log')
logger.addHandler(file_log_handler)

stderr_log_handler = logging.StreamHandler()
logger.addHandler(stderr_log_handler)

# nice output format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_log_handler.setFormatter(formatter)
stderr_log_handler.setFormatter(formatter)

logger.setLevel( 'DEBUG' )

logger.info('Args: {}'.format(args))
logger.info('Exp_name = ' + exp_name)
logger.info('Prefix = ' + prefix)
args.prefix = prefix

torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.device = device

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.manual_seed(args.seed)

if args.dataset == 'Add-Task': print('Add task')
input_size = 2

optimizer = None
lr = args.lr


rnn = nn.LSTM( input_size, args.nhid )
model = AddTaskModel( rnn, args.nhid )
total_params = ptb_count_parameters(model)


if len(args.load) > 0:
    logger.info("Loaded model\n")
    model_ckp = torch.load(args.load)
    model.load_state_dict(model_ckp['state_dict'])
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr, weight_decay=args.wdecay)
    optimizer.load_state_dict(model_ckp['optimizer'])
    print('best acc of loaded model: ',model_ckp['best_acc1'])

print('Model: ',model)
if args.cuda:
    model.cuda()

if optimizer is None:
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr, weight_decay=args.wdecay)
    if args.optim == 'SGD':
        optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.wdecay)


all_test_losses = []
epochs = args.epochs #100
best_acc1 = 0.0
best_val_loss = None
first_update = False
named_params = []#get_stats_named_params( model )
for epoch in range(1, epochs + 1):
    start = time.time()
    all_test_losses.append(add_task_train_online( model, optimizer, args, named_params,logger ))
    break
# print(all_test_losses)

with open('your_lstm_bptt.txt', 'w') as f:
    for item in all_test_losses:
        f.write("%s\n" % item)

import matplotlib.pyplot as plt
plt.plot(all_test_losses[0])
# plt.ylim(0,0.2)
plt.show()