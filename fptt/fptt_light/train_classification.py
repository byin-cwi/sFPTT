
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


from utils import get_xt,update_prob_estimates
# from snn_models_LIF4 import *
# from snn_models_LIF5 import *
from datasets import data_generator, adding_problem_generator

def save_checkpoint(state, is_best, prefix, filename='_snn_checkpoint.pth.tar'):
    print('saving at ', prefix+filename)
    torch.save(state, prefix+filename)
    if is_best:
        shutil.copyfile(prefix+filename, prefix+ '_snn_model_best.pth.tar')

def get_stats_named_params(model):
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

def test(model, test_loader, logger,args,permute=None):
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        b,input_channels,d1,d2 = data.shape
        seq_length = d1*d2
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = data.view(-1, input_channels, seq_length)
        if args.permute:
            data = data[:, :, permute]

        hidden = model.init_hidden(data.size(0))
        
        outputs, hidden, recon_loss = model(data, hidden)        
        output = outputs[-1]
        test_loss += F.nll_loss(output, target, reduction='sum').data.item()
        pred = output.data.max(1, keepdim=True)[1]
        
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
           test_loss, correct, len(test_loader.dataset),
           100. * correct / len(test_loader.dataset)))
    sys.stdout.flush()
    return test_loss, 100. * correct / len(test_loader.dataset)


def train(args, train_loader,test_loader, permute, n_classes, 
            model, logger, optimizer,scheduler=None):
    steps = 0
    estimate_class_distribution = torch.zeros(n_classes, args.parts, n_classes, dtype=torch.float)
    estimatedDistribution = None
    if args.per_ex_stats:
        estimatedDistribution = torch.zeros(len(train_loader)*args.batch_size, args.parts, n_classes, dtype=torch.float)

    batch_size = args.batch_size
    alpha = args.alpha
    beta = args.beta
    all_test_losses = []
    epochs = args.epochs 
    best_acc1 = 0.0
    best_val_loss = None
    first_update = False
    named_params = get_stats_named_params( model )

    PARTS = args.parts
    for epoch in range(1,1+epochs):
        start = time.time()
        if args.per_ex_stats and epoch%5 == 1 :
            first_update = update_prob_estimates( model, args, train_loader, permute, estimatedDistribution, estimate_class_distribution, first_update )

        
        train_loss = 0
        total_clf_loss = 0
        total_regularizaton_loss = 0
        total_oracle_loss = 0
        model.train()
        

        for batch_idx, (data, target) in enumerate(train_loader):
            b,input_channels,d1,d2 = data.shape
            seq_length = d1*d2

            if args.cuda: data, target = data.cuda(), target.cuda()
            data = data.view(-1, input_channels, seq_length)
    
    
            if args.permute:
                data = data[:, :, permute]
            
            B = target.size()[0]
            step = model.network.step
            xdata = data.clone()
            pdata = data.clone()
            
            inputs = xdata.permute(2, 0, 1) 
            T = inputs.size()[0]
    
            Delta = torch.zeros(B, dtype=xdata.dtype, device=xdata.device)
            
            _PARTS = PARTS
            if (PARTS * step < T):
                _PARTS += 1
            for p in range(_PARTS):
                x, start, end = get_xt(p, step, T, inputs)
                
                if p==0:
                    h = model.init_hidden(xdata.size(0))
                else:
                    h = tuple(v.detach() for v in h)
                
                if p<PARTS-1:
                    if epoch < 20:
                        if args.per_ex_stats:
                            oracle_prob = estimatedDistribution[batch_idx*batch_size:(batch_idx+1)*batch_size, p]
                        else:
                            oracle_prob = 0*estimate_class_distribution[target, p] + (1.0/n_classes)
                    else:
                        oracle_prob = estimate_class_distribution[target, p]
                else:
                    oracle_prob = F.one_hot(target).float() 

                
                o, h,hs = model.network.forward(x, h )
                # print(os[-1].shape,h[-1].shape,hs[-1][-1].shape)
                # print(h[-1],os[-1])
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
                
                clf_loss = (p+1)/(_PARTS)*F.nll_loss(output, target)
                # clf_loss = F.nll_loss(output, target)
                # clf_loss = (p+1)/(_PARTS)*F.gaussian_null_loss(output, target)
                oracle_loss = (1 - (p+1)/(_PARTS)) * 1.0 *torch.mean( -oracle_prob * output )
                    
                regularizer = get_regularizer_named_params( named_params, args, _lambda=1.0 )      
                loss = clf_loss + oracle_loss + regularizer #+ model.network.fr*0.5

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

            
            lr = optimizer.param_groups[0]['lr']

            steps += T#seq_length
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

        if scheduler is not None:
                scheduler.step()

        reset_named_params(named_params, args)

        test_loss, acc1 = test( model, test_loader, logger,args,permute)
        logger.info('time taken = ' + str(time.time() - start) )

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
            
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                #'oracle_state_dict': oracle.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                #'oracle_optimizer' : oracle_optim.state_dict(),
            }, is_best, prefix=args.prefix)
 
        all_test_losses.append(test_loss)
    return all_test_losses

        