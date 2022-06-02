"""
https://github.com/bubbliiiing/yolov4-tiny-pytorch/blob/36cd4fc74a1c251ed45e81008a240baeb752f73d/nets/yolo_training.py
"""
import os
import shutil
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
import math

###############################################################################################
###############################    Define SNN layer   #########################################
###############################################################################################

b_j0 = 0.1  # neural threshold baseline
R_m = 3  # membrane resistance
dt = 1  
gamma = .5  # gradient scale
lens = 0.3

def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma


class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold
        ctx.save_for_backward(input)
        return input.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # temp = abs(input) < lens
        scale = 6.0
        hight = .15
        # temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(math.pi))/lens
        temp = gaussian(input, mu=0., sigma=lens) * (1. + hight) \
               - gaussian(input, mu=lens, sigma=scale * lens) * hight \
               - gaussian(input, mu=-lens, sigma=scale * lens) * hight
        # temp =  gaussian(input, mu=0., sigma=lens)
        return grad_input * temp.float() * gamma


act_fun_adp = ActFun_adp.apply

def mem_update_adp(inputs, mem, spike, tau_m, dt=1, isAdapt=1):
    alpha = tau_m
    
    # ro = tau_adp

    # if isAdapt:
    #     beta = 1.8
    # else:
    #     beta = 0.
    # # print('ro: ',ro.shape, '; b:', b.shape, '; spk: ',spike.shape)
    # b = ro * b + (1 - ro) * spike
    # B = b_j0 + beta * b
    B = 1.

    d_mem = -mem + inputs
    mem = mem + d_mem*alpha
    inputs_ = mem - B

    spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
    mem = (1-spike)*mem

    return mem, spike


def output_Neuron(inputs, mem, tau_m, dt=1):
    """
    The read out neuron is leaky integrator without spike
    """
    d_mem = -mem  +  inputs
    mem = mem+d_mem*tau_m
    return mem

class MaxPoolStride1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x
###############################################################################################
###############################################################################################
###############################################################################################

class SNN_dense_cell(nn.Module):
    def __init__(self, input_size, hidden_size, is_rec=False):
        super(SNN_dense_cell, self).__init__()
        print('SNN-ltc ')
    
        
        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.is_rec = is_rec
        
        self.rnn_name = 'SNN-ltc cell'

        if is_rec:
            self.layer1_x = nn.Linear(input_size+hidden_size, hidden_size)
        else:
            self.layer1_x = nn.Linear(input_size, hidden_size)
        self.layer1_tauM = nn.Linear(hidden_size+hidden_size, hidden_size)
        # self.layer1_tauAdp = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.act1 = nn.Sigmoid()

    def forward(self, x_t, mem_t,spk_t):    
        if self.is_rec:
            dense_x = self.layer1_x(torch.cat((x_t,spk_t),dim=-1))
        else:
            dense_x = self.layer1_x(x_t)

        tauM1 = self.act1(self.layer1_tauM(torch.cat((dense_x, mem_t),dim=-1)))
        # tauAdp1 = self.act1(self.layer1_tauAdp(torch.cat((dense_x,b_t),dim=-1)))

        mem_1,spk_1,_ = mem_update_adp(dense_x, mem=mem_t,spike=spk_t,
                                        tau_m=tauM1)

        return mem_1,spk_1

    def compute_output_size(self):
        return [self.hidden_size]


class SNN_Conv_cell(nn.Module):
    def __init__(self, input_size, output_dim, kernel_size=5,strides=1,padding=0,
                 pooling_type = None, pool_size = 2, pool_strides =2,bias=True):
        super(SNN_Conv_cell, self).__init__()
        
        print('SNN-conv +', pooling_type)
        self.input_size = input_size
        self.input_dim = input_size[0]
        self.output_dim = output_dim
    
        self.input_size = input_size
        
        
        self.rnn_name = 'SNN-conv cell'
        if pooling_type is not None: 
            if pooling_type =='max':
                self.pooling = nn.MaxPool2d(kernel_size=pool_size, stride=pool_strides, padding=0)
            elif pooling_type =='avg':
                self.pooling = nn.AvgPool2d(kernel_size=pool_size, stride=pool_strides, padding=0)
            elif pooling_type =='maxS1':
                self.pooling = MaxPoolStride1()
            elif pooling_type =='up':
                self.pooling = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.pooling = None

        self.conv1_x = nn.Conv2d(self.input_dim,output_dim,kernel_size=kernel_size,stride=strides,padding=kernel_size//2,bias=bias)
        self.conv_tauM = nn.Conv2d(output_dim,output_dim,kernel_size=3,stride=1,padding=1)

        self.sig1 = nn.Sigmoid()
     
        self.BN = nn.BatchNorm2d(output_dim)
        self.BN1 = nn.BatchNorm2d(output_dim)

        self.output_size = self.compute_output_size()

        # nn.init.kaiming_normal_(self.conv1_x.weight)
        # nn.init.kaiming_normal_(self.conv_tauM.weight)
        # nn.init.kaiming_normal_(self.conv_tauAdp.weight)
        nn.init.xavier_normal_(self.conv1_x.weight)
        nn.init.xavier_normal_(self.conv_tauM.weight)
        # nn.init.xavier_normal_(self.conv_tauAdp.weight)
        if bias:
            nn.init.constant_(self.conv1_x.bias,0)
            nn.init.constant_(self.conv_tauM.bias,0)


    def forward(self, x_t, mem_t,spk_t, short_cut = None):
        conv_bnx = self.BN(self.conv1_x(x_t.float()))

        if short_cut is not None:
            conv_bnx  = conv_bnx + short_cut
    
        if self.pooling is not None: 
            conv_x = self.pooling(conv_bnx)
        else:
            conv_x = conv_bnx

        tauM1 = self.sig1(self.BN1(self.conv_tauM(conv_x+mem_t)))
        # tauAdp1 = self.sig3(self.BN2(self.conv_tauAdp(conv_x+b_t)))
        mem_1,spk_1 = mem_update_adp(conv_x, mem=mem_t,spike=spk_t,
                                        tau_m=tauM1)

        return mem_1,spk_1,conv_bnx

    def compute_output_size(self):
        x_emp = torch.randn([1,self.input_size[0],self.input_size[1],self.input_size[2]])   
        out = self.conv1_x(x_emp)
        if self.pooling is not None: 
            out=self.pooling(out)
        return out.shape[1:]

'''
                    input
                      |
                  BasicConv
                      -----------0-----------
                      |1                    |
                 route_group              route
                      |                     |
                  BasicConv                 |
                      |                     |
    -------------------                     |
    |                 |                     |
 route_1          BasicConv                 |
    |                 |                     |
    -----------------cat                    |
                      |                     |
        ----      BasicConv                 |
        |             |                     |
      feat           cat---------------------
                      |
                 MaxPooling2D
'''

class BasicBlock(nn.Module):
    def __init__(self,in_size,planes,stride=1):
        super(BasicBlock,self).__init__()
        self.network = []
        self.network_size = []
        self.planes = planes
        biased = False

        self.maxpool = nn.MaxPool2d([2,2],[2,2])

        self.conv1 = SNN_Conv_cell(in_size,planes,3,stride,1,bias=biased)
        in_size1 = self.conv1.compute_output_size()

        self.conv2 = SNN_Conv_cell([planes//2,in_size1[1],in_size1[2]], planes//2,3,1,1,bias=biased)
        in_size2 = self.conv2.compute_output_size()

        self.conv3 = SNN_Conv_cell(in_size2,planes//2,3,1,1,bias=biased)
        in_size3 = self.conv3.compute_output_size()

        self.conv4 = SNN_Conv_cell([planes,in_size3[1],in_size3[2]],planes,1,1,0,bias=biased)
        in_size4 = self.conv4.compute_output_size()

        self.network_size = [in_size1,in_size2,in_size3,in_size4]
        self.network = [self.conv1, self.conv2,self.conv3, self.conv4]


    def forward(self,x,h,layer_idx, fr):
        layer_i = layer_idx +1

        h[2*layer_i],h[1+2*layer_i],_  = self.conv1(x,h[2*layer_i],h[1+2*layer_i])
        x_in = h[1+2*layer_i]
        fr += x_in.detach().cpu().numpy().mean()

        route = x_in
        x_in = torch.split(x_in, self.planes//2, dim = 1)[1]

        layer_i = layer_i +1

        h[2*layer_i],h[1+2*layer_i],_  = self.conv2(x_in,h[2*layer_i],h[1+2*layer_i])
        x_in = h[1+2*layer_i]
        fr += x_in.detach().cpu().numpy().mean()
        
        route1 = x_in

        layer_i = layer_i +1

        h[2*layer_i],h[1+2*layer_i],_  = self.conv3(x_in,h[2*layer_i],h[1+2*layer_i])
        x_in = h[1+2*layer_i]
        fr += x_in.detach().cpu().numpy().mean()    

        x_in =  torch.cat([x_in,route1], dim = 1)   

        layer_i = layer_i +1

        h[2*layer_i],h[1+2*layer_i],_  = self.conv4(x_in,h[2*layer_i],h[1+2*layer_i])
        x_in = h[1+2*layer_i]
        fr += x_in.detach().cpu().numpy().mean()

        feat= x_in

        x_in = torch.cat([route, x_in], dim = 1)

        x_in = self.maxpool(x_in)

        return x_in,feat, layer_i,h, fr

