
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
from datasets import data_generator, adding_problem_generator

def add_task_train_online( net, optimizer, args, named_params, logger):
    batch_size = args.batch_size
    n_steps = args.epochs
    c_length = args.bptt

    losses = []
    
    PARTS = args.parts #10
    step = c_length // PARTS
    logger.info('step = ' + str(step))
    
    alpha = 0.05 #0.2
    alpha1 = 0.005 #001
    alpha2 = 0.01

    # torch.autograd.set_detect_anomaly(True)
    for i in range(n_steps):        
        s_t = time.time()
        x,y = adding_problem_generator(batch_size, seq_len=c_length, number_of_ones=2)        
        x = x.cuda()
        y = y.cuda()
        data = x.transpose(0, 1)
        y = y.transpose(0, 1)
        
        net.train()
        xdata = data.clone()
        inputs = xdata
        
        T = c_length
        
        # h = net.init_hidden(batch_size)               

        # optimizer.zero_grad()
        # # print(x.shape)
        # loss, h = net.forward(inputs, y, h) 
        # # loss = (p+1/PARTS) *  loss
        
        # loss.backward()
        
        # torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        # optimizer.step()

        
        optimizer.zero_grad()
        # h = get_initial_hidden_state(net, batch_size, hidden_size)  
        h = net.init_hidden(batch_size)              
        x = data
        loss, _ = net.forward(x, y, h)
        loss_act = loss
        loss.backward()        
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        optimizer.step()
        
        
        
        ### Evaluate
        net.eval()
        x,y = adding_problem_generator(batch_size, seq_len=c_length, number_of_ones=2)        
        x = x.cuda()
        y = y.cuda()
        x = x.transpose(0, 1)
        y = y.transpose(0, 1)

        h = net.init_hidden(batch_size)               
        loss, _ = net.forward(x, y, h)
        loss_act = loss
        losses.append(loss_act.item())

        if i%args.log_interval == 0:
            logger.info('Update {}, Time for Update: {} , Average Loss: {}'
                  .format(i +1, time.time()- s_t, loss_act.item() ))
    
    logger.info("Average loss: " + str(np.mean(np.array(losses))) )
    logger.info('Losses : ' + str( losses ))
    return losses