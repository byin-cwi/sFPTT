import os
import shutil
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from snn_models_LIF4 import *


class AddTaskModel(nn.Module):
    def __init__(self, input_size,hidden_size):
        super(AddTaskModel, self).__init__()
        self.nhid = hidden_size
        
        self.layer1_x = nn.Linear(input_size+hidden_size, hidden_size)
        self.layer1_tauM = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.layer1_tauAdp = nn.Linear(hidden_size+hidden_size, hidden_size)

        self.lin = nn.Linear(hidden_size, 1) 
        self.loss_func = nn.MSELoss()

        nn.init.xavier_normal_(self.lin.weight)

        self.act1 = nn.Sigmoid()
        self.act2 = nn.Sigmoid()

    def forward(self, x, y, h):        
        loss = 0
        # print(x.shape,y.shape)
        s,b,d = x.shape
        spk_sum = 0
        out = 0
        for i in range(s):
            dense_x = self.layer1_x(torch.cat((x[i,:,:],h[1]),dim=-1))
            tauM1 = self.act1(self.layer1_tauM(torch.cat((dense_x, h[0]),dim=-1)))
            tauAdp1 = self.act2(self.layer1_tauAdp(torch.cat((dense_x, h[2]),dim=-1)))

            mem_1,spk_1,_,b_1 = mem_update_adp(dense_x, mem=h[0],spike=h[1],
                                            tau_adp=tauAdp1,tau_m=tauM1,b = h[2])

            h = (mem_1,spk_1,b_1,out)
            # output, hidden = self.rnn.forward(x, h)
            # spk_sum  = spk_sum + spk_1
        # out = self.lin(spk_sum/s)   
        out = self.lin(mem_1)           
        out = out.squeeze()
        y  = y.squeeze()
        # print(y.shape)
        loss += self.loss_func(out, y.t())

        h = (mem_1,spk_1,b_1,out)
        return loss, h

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(bsz, self.nhid),
                weight.new_zeros(bsz, self.nhid),
                weight.new_zeros(bsz, self.nhid),
                weight.new_zeros(bsz, self.nhid),)


# def mem_update_adp(inputs, mem, spike, tau_adp,tau_m, b, dt=1, isAdapt=1):
#     alpha = tau_m#torch.exp(-1. * dt / tau_m).cuda()
#     ro = tau_adp#torch.exp(-1. * dt / tau_adp).cuda()
#     # tau_adp is tau_adaptative which is learnable # add requiregredients
#     if isAdapt:
#         beta = 1.8
#     else:
#         beta = 0.

#     b = ro * b + (1 - ro) * spike
#     B = b_j0 + beta * b

#     mem = mem * alpha + (1 - alpha) * R_m * inputs - B * spike * dt
#     inputs_ = mem - B
#     spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
#     return mem, spike, B, b


# class AddTaskModel(nn.Module):
#     def __init__(self, input_size,hidden_size):
#         super(AddTaskModel, self).__init__()
#         self.nhid = hidden_size
        
#         self.layer1_x = nn.Linear(input_size+hidden_size, hidden_size)
#         self.tau_adp = nn.Parameter(torch.Tensor(hidden_size))
#         self.tau_m = nn.Parameter(torch.Tensor(hidden_size))

#         self.lin = nn.Linear(hidden_size, 1) 
#         self.loss_func = nn.MSELoss()

#         nn.init.xavier_normal_(self.lin.weight)
#         # nn.init.normal_(self.tau_adp, 200,50)
#         # nn.init.normal_(self.tau_m, 20,5)
#         nn.init.normal_(self.tau_adp, 4,1.)
#         nn.init.normal_(self.tau_m, 2,1.)

#         self.act1 = nn.Sigmoid()
#         self.act2 = nn.Sigmoid()

#     def forward(self, x, y, h):        
#         loss = 0
#         # print(x.shape,y.shape)
#         s,b,d = x.shape
#         spk_sum = 0
#         out = 0
#         for i in range(s):
#             dense_x = self.layer1_x(torch.cat((x[i,:,:],h[1]),dim=-1))
#             # tauM1 = self.act1(self.layer1_tauM(torch.cat((dense_x, h[0]),dim=-1)))
#             # tauAdp1 = self.act2(self.layer1_tauAdp(torch.cat((dense_x, h[2]),dim=-1)))

#             # mem_1,spk_1,_,b_1 = mem_update_adp(dense_x, mem=h[0],spike=h[1],
#             #                                 tau_adp=torch.exp(-1./self.tau_adp),tau_m=torch.exp(-1./self.tau_m),b = h[2])
#             mem_1,spk_1,_,b_1 = mem_update_adp(dense_x, mem=h[0],spike=h[1],
#                                             tau_adp=self.act1(self.tau_adp),tau_m=self.act1(self.tau_m),b = h[2])

#             h = (mem_1,spk_1,b_1,out)
#             # output, hidden = self.rnn.forward(x, h)
#             # spk_sum  = spk_sum + spk_1
#         # out = self.lin(spk_sum/s)   
#         out = self.lin(mem_1)           
#         out = out.squeeze()
#         y  = y.squeeze()
#         # print(y.shape)
#         loss += self.loss_func(out, y.t())

#         h = (mem_1,spk_1,b_1,out)
#         return loss, h

#     def init_hidden(self, bsz):
#         weight = next(self.parameters())
#         return (weight.new_zeros(bsz, self.nhid),
#                 weight.new_zeros(bsz, self.nhid),
#                 weight.new_zeros(bsz, self.nhid),
#                 weight.new_zeros(bsz, self.nhid),)