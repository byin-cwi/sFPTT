
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

from snn_models_LIF4_cnn import *

# from snn_models_LIF_dvs_vggv1 import *
# from snn_models_LIF4_vgg import *
# from snn_models_LIF_dvs_vgg import *

from snn_models_LIF_res import *
from snn_models_LIF_cifarNet import *
from datasets import data_generator, adding_problem_generator


def get_stats_named_params( model ):
    named_params = {}
    for name, param in model.named_parameters():
        sm, lm, dm = param.detach().clone(), 0.0*param.detach().clone(), 0.0*param.detach().clone()
        named_params[name] = (param, sm, lm, dm)
    return named_params

def pre_pre_optimizer_updates( named_params, args ):
    if not args.debias: return
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        param_data = param.data.clone()
        param.data.copy_( sm.data )
        sm.data.copy_( param_data )
        del param_data

def pre_optimizer_updates( named_params, args ):
    if not args.debias: return
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        lm.data.copy_( param.grad.detach() )
        param_data = param.data.clone()
        param.data.copy_( sm.data )
        sm.data.copy_( param_data )
        del param_data

def post_optimizer_updates( named_params, args, epoch ):
    alpha = args.alpha
    beta = args.beta
    rho = args.rho
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        if args.debias:
            beta = (1. / (1. + epoch))
            sm.data.mul_( (1.0-beta) )
            sm.data.add_( beta * param )

            rho = (1. / (1. + epoch))
            dm.data.mul_( (1.-rho) )
            dm.data.add_( rho * lm )
        else:
            lm.data.add_( -alpha * (param - sm) )
            sm.data.mul_( (1.0-beta) )
            sm.data.add_( beta * param - (beta/alpha) * lm )

def get_regularizer_named_params( named_params, args, _lambda=1.0 ):
    alpha = args.alpha
    rho = args.rho
    regularization = torch.zeros( [], device=args.device )
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        regularization += (rho-1.) * torch.sum( param * lm )
        if args.debias:
            regularization += (1.-rho) * torch.sum( param * dm )
        else:
            r_p = _lambda * 0.5 * alpha * torch.sum( torch.square(param - sm) )
            regularization += r_p
            # print(name,r_p)
    return regularization 

def reset_named_params(named_params, args):
    if args.debias: return
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        param.data.copy_(sm.data)

def test(model, test_loader, logger):
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = data.view(-1, 3,32,32)
        with torch.no_grad():
            model.eval()
            hidden = model.init_hidden(data.size(0))
        
            outputs, hidden, recon_loss = model(data, hidden) 
           
            output = outputs[-1]
            test_loss += F.nll_loss(output, target, reduction='sum').data.item()
            pred = output.data.max(1, keepdim=True)[1]
        
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        torch.cuda.empty_cache()

    test_loss /= len(test_loader.dataset)
    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
           test_loss, correct, len(test_loader.dataset),
           100. * correct / len(test_loader.dataset)))
    sys.stdout.flush()
    return test_loss, 100. * correct / len(test_loader.dataset)


def train(epoch, args, train_loader, permute, n_classes, model, named_params, logger):
    global steps
    global estimate_class_distribution

    batch_size = args.batch_size
    alpha = args.alpha
    beta = args.beta

    PARTS = args.parts
    train_loss = 0
    total_clf_loss = 0
    total_regularizaton_loss = 0
    total_oracle_loss = 0
    model.train()
    

    T = PARTS#40
    # PARTS = T
    #entropy = EntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda: data, target = data.cuda(), target.cuda()
        # print(data.shape)
        data = data.view(-1, 3,32,32 )
       
        B = target.size()[0]
        step = model.network.step
        # xdata = data.clone()
        
        # inputs = data
        
        Delta = torch.zeros(B, dtype=data.dtype, device=data.device)
        
        _PARTS = PARTS
        for p in range(_PARTS):
            if p==0:
                h = model.init_hidden(data.size(0))
            else:
                h = tuple(v.detach() for v in h)
            # print([p.shape for p in h])
            if p<PARTS-1:
                if epoch < 10:
                    if args.per_ex_stats:
                        oracle_prob = estimatedDistribution[batch_idx*batch_size:(batch_idx+1)*batch_size, p]
                    else:
                        oracle_prob = 0*estimate_class_distribution[target, p] + (1.0/n_classes)
                else:
                    oracle_prob = estimate_class_distribution[target, p]
            else:
                oracle_prob = F.one_hot(target,n_classes).float() 

            
            o, h,hs = model.network.forward(data, h )
            # print(os[-1].shape,h[-1].shape,hs[-1][-1].shape)
            # print(h[-1],os[-1])
            # print(x.shape)
            # print(os.shape)

            prob_out = F.softmax(h[-1], dim=1)
            output = F.log_softmax(h[-1], dim=1) 

            if p<PARTS-1:
                with torch.no_grad():
                    filled_class = [0]*n_classes
                    n_filled = 0
                    for j in range(B):
                        if n_filled==n_classes: break

                        y = target[j].item()
                        if filled_class[y] == 0 and (torch.argmax(prob_out[j]) != target[j]):
                            filled_class[y] = 1
                            estimate_class_distribution[y, p] = prob_out[j].detach()
                            n_filled += 1

            optimizer.zero_grad()
            
            # clf_loss = F.nll_loss(output, target)
            # clf_loss = F.nll_loss(output, target)
            clf_loss = F.cross_entropy(output, target)
            oracle_loss = (1 - (p+1)/(_PARTS)) * 1.0 *torch.mean( -oracle_prob * output )
                
            regularizer = get_regularizer_named_params( named_params, args, _lambda=1.0 )      
            loss = clf_loss + oracle_loss + regularizer + model.network.fr*0.05

            # loss.backward(retain_graph=True)
            loss.backward()
            # print(model.network.layer1_x.weight.grad, model.network.tau_m_r1.grad)
            # print(os.shape)

            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                
                
            optimizer.step()
            post_optimizer_updates( named_params, args,epoch )
        
            train_loss += loss.item()
            total_clf_loss += clf_loss.item()
            total_regularizaton_loss += regularizer #.item()
            total_oracle_loss += oracle_loss.item()
        
        steps += seq_length
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr: {:.6f}\tLoss: {:.6f}\tOracle: \
                {:.6f}\tClf: {:.6f}\tReg: {:.6f}\tFr: {:.6f}\tSteps: {}'.format(
                   epoch, batch_idx * batch_size, len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), lr, train_loss / args.log_interval, 
                   total_oracle_loss / args.log_interval, 
                   total_clf_loss / args.log_interval, total_regularizaton_loss / args.log_interval, model.network.fr,steps))
            # print(model.network.fr)
            train_loss = 0
            total_clf_loss = 0
            total_regularizaton_loss = 0
            total_oracle_loss = 0

            sys.stdout.flush()
    # print(model.network.layer1_x.weight.grad, model.network.tau_m_r1.grad)
    # print( model.network.tau_m_r1.grad)





parser = argparse.ArgumentParser(description='Sequential Decision Making..')

parser.add_argument('--alpha', type=float, default=.1, help='Alpha')
parser.add_argument('--beta', type=float, default=0.5, help='Beta')
parser.add_argument('--rho', type=float, default=0.0, help='Rho')
parser.add_argument('--lmbda', type=float, default=2.0, help='Lambda')
parser.add_argument('--debias', action='store_true', help='FedDyn debias algorithm')
parser.add_argument('--K', type=int, default=1, help='Number of iterations for debias algorithm')

parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=256, help='size of word embeddings')
parser.add_argument('--nlayers', type=int, default=1, #2,
                    help='number of layers')
parser.add_argument('--bptt', type=int, default=300, #35,
                    help='sequence length')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')

parser.add_argument('--n_experts', type=int, default=15,
                    help='PTB-Word n_experts')
parser.add_argument('--nhid', type=int, default=256,
                    help='number of hidden units per layer')
parser.add_argument('--nhidlast', type=int, default=620,
                    help='number of hidden units per layer')
parser.add_argument('--lr', type=float, default=5e-3,
                    help='initial learning rate (default: 4e-3)')
parser.add_argument('--clip', type=float, default=1., #0.5,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit (default: 200)')
parser.add_argument('--parts', type=int, default=10,
                    help='Parts to split the sequential input into (default: 10)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--small_batch_size', type=int, default=-1, metavar='N',
                    help='batch size')
parser.add_argument('--max_seq_len_delta', type=int, default=40, metavar='N',
                    help='batch size')  
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                    help='batch size')

parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='output locked dropout (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.2,
                    help='input locked dropout (0 = no dropout)')
parser.add_argument('--dropoutl', type=float, default=0.29,
                    help='input locked dropout (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.1,
                    help='dropout applied to weights (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.2,
                    help='dropout applied to hidden layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
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
parser.add_argument('--when', nargs='+', type=int, default=[10,20,40,80,120,150],#[30,70,120],#[10,20,50, 75, 90],
                    help='When to decay the learning rate')
parser.add_argument('--load', type=str, default='',
                    help='path to load the model')
parser.add_argument('--save', type=str, default='./models/',
                    help='path to load the model')

parser.add_argument('--per_ex_stats', action='store_true',
                    help='Use per example stats to compute the KL loss (default: False)')
parser.add_argument('--permute', action='store_true',
                    help='use permuted dataset (default: False)')
parser.add_argument('--dataset', type=str, default='CIFAR-10',
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




torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.device = device

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.manual_seed(args.seed)



steps = 0
if args.dataset in ['CIFAR-10', 'MNIST-10','FMNIST']:
    train_loader, test_loader, seq_length, input_channels, n_classes = data_generator(args.dataset, 
                                                                     batch_size=args.batch_size,
                                                                     dataroot=args.dataroot, 
                                                                     shuffle=(not args.per_ex_stats))
    permute = torch.Tensor(np.random.permutation(seq_length).astype(np.float64)).long()   # Use only if args.permute is True

    estimate_class_distribution = torch.zeros(n_classes, args.parts, n_classes, dtype=torch.float)
    estimatedDistribution = None
    if args.per_ex_stats:
        estimatedDistribution = torch.zeros(len(train_loader)*args.batch_size, args.parts, n_classes, dtype=torch.float)
else:
    logger.info('Unknown dataset.. customize the routines to include the train/test loop.')
    exit(1)

optimizer = None
lr = args.lr


model = SeqModel(ninp=[3,32,32],
                    nhid=args.nhid,
                    nout=n_classes,
                    dropout=args.dropout,
                    dropouti=args.dropouti,
                    dropouth=args.dropouth,
                    wdrop=args.wdrop,
                    temporalwdrop=args.temporalwdrop,
                    wnorm=args.wnorm,
                    n_timesteps=seq_length, 
                    parts=args.parts)

total_params = count_parameters(model)
if args.cuda:
    permute = permute.cuda()
if len(args.load) > 0:
    logger.info("Loaded model\n")
    model_ckp = torch.load(args.load)
    model.load_state_dict(model_ckp['state_dict'])
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr, weight_decay=args.wdecay)
    optimizer.load_state_dict(model_ckp['optimizer'])
    print('best acc of loaded model: ',model_ckp['best_acc1'])

# print('Model: ',model)
if args.cuda:
    model.cuda()


if optimizer is None:

    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr, weight_decay=args.wdecay)
    if args.optim == 'SGD':
        optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.wdecay)
        
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=30)

logger.info('Optimizer = ' + str(optimizer) )
logger.info('Model total parameters: {}'.format(total_params))

all_test_losses = []
epochs = args.epochs #100
best_acc1 = 0.0
best_val_loss = None
first_update = False
named_params = get_stats_named_params( model )
for epoch in range(1, epochs + 1):
    start = time.time()
    
    if args.dataset in ['CIFAR-10', 'MNIST-10']:
        if args.per_ex_stats and epoch%5 == 1 :
            first_update = update_prob_estimates( model, args, train_loader, permute, estimatedDistribution, estimate_class_distribution, first_update )

        train(epoch, args, train_loader, permute, n_classes, model, named_params, logger)   
        #train_oracle(epoch)

        reset_named_params(named_params, args)

        test_loss, acc1 = test( model, test_loader, logger )
    
        logger.info('time taken = ' + str(time.time() - start) )
        scheduler.step()
        # if epoch in args.when:
        #     # Scheduled learning rate decay
        #     lr /= 2.
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr

        # linear lr decay
        # lr = lr*(1-1./epochs)+1e-6
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr
        # if epoch>0 and epoch%20==0 and epoch<100:
        #     # Scheduled learning rate decay
        #     lr /= 5.
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
            
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
            
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                #'oracle_state_dict': oracle.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                #'oracle_optimizer' : oracle_optim.state_dict(),
            }, is_best, prefix=prefix)
 
        all_test_losses.append(test_loss)

test_loss, acc1 = test( model, test_loader, logger )