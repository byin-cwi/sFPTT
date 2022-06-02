"""
Liquid time constant snn
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
        # return grad_input


act_fun_adp = ActFun_adp.apply

def mem_update_adp(inputs, mem, spike, tau_m, dt=1):
    alpha = tau_m
    B = .5


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
###############################################################################################
###############################################################################################
###############################################################################################

class SNN_Conv_cell(nn.Module):
    def __init__(self, input_size, output_dim, kernel_size=5,strides=1,padding=0,
                 pooling_type = None,pool_size = 2, pool_strides =2):
        super(SNN_Conv_cell, self).__init__()
        
        # print('SNN-conv ')
        self.input_size = input_size
        self.input_dim = input_size[0]
        self.output_dim = output_dim
    
        self.input_size = input_size
        
        
        self.rnn_name = 'SNN-conv cell'
        if pooling_type is not None: 
            if pooling_type =='max':
                self.pooling = nn.MaxPool2d(kernel_size=pool_size, stride=pool_strides, padding=1)
            elif pooling_type =='avg':
                self.pooling = nn.AvgPool2d(kernel_size=pool_size, stride=pool_strides, padding=1)
        else:
            self.pooling = None

        self.conv1_x = nn.Conv2d(self.input_dim,output_dim,kernel_size=kernel_size,stride=strides,padding=padding)
        self.conv_tauM = nn.Conv2d(output_dim,output_dim,kernel_size=kernel_size,stride=strides,padding=padding)
        self.sig1 = nn.Sigmoid()
        self.BN = nn.BatchNorm2d(output_dim)
        self.BN1 = nn.BatchNorm2d(output_dim)

        self.output_size = self.compute_output_size()

        nn.init.xavier_uniform_(self.conv1_x.weight)
        nn.init.xavier_uniform_(self.conv_tauM.weight)
        nn.init.constant_(self.conv1_x.bias,0)
        nn.init.constant_(self.conv_tauM.bias,0)

    def forward(self, x_t, mem_t,spk_t):

        conv_x = self.BN(self.conv1_x(x_t.float()))
        if self.pooling is not None: 
            conv_x = self.pooling(conv_x)

        tauM1 = self.sig1(self.BN1(self.conv_tauM(conv_x+mem_t)))
        # tauM1 = self.sig1(self.conv_tauM(conv_x+mem_t))

        mem_1,spk_1= mem_update_adp(conv_x, mem=mem_t,spike=spk_t,tau_m=tauM1)

        return mem_1,spk_1

    def compute_output_size(self):
        x_emp = torch.randn([1,self.input_size[0],self.input_size[1],self.input_size[2]])   
        out = self.conv1_x(x_emp)
        if self.pooling is not None: 
            out=self.pooling(out)
        return out.shape[1:]

class SNN_rec_cell(nn.Module):
    def __init__(self, input_size, hidden_size,is_rec = True):
        super(SNN_rec_cell, self).__init__()
    
        
        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.is_rec = is_rec

        if is_rec:
            self.layer1_x = nn.Linear(input_size+hidden_size, hidden_size)
        else:
            self.layer1_x = nn.Linear(input_size, hidden_size)
        self.layer1_tauM = nn.Linear(2*hidden_size, hidden_size)

        self.act1 = nn.Sigmoid()

        nn.init.xavier_uniform_(self.layer1_x.weight)
        nn.init.xavier_uniform_(self.layer1_tauM.weight)
        nn.init.constant_(self.layer1_x.bias,0)
        nn.init.constant_(self.layer1_tauM.bias,0)


    def forward(self, x_t, mem_t,spk_t):    

        if self.is_rec:
            # print(x_t.shape, spk_t.shape)
            dense_x = self.layer1_x(torch.cat((x_t,spk_t),dim=-1))
        else:
            dense_x = self.layer1_x(x_t)
        tauM1 = self.act1(self.layer1_tauM(torch.cat((dense_x,mem_t),dim=-1)))
    
        mem_1,spk_1 = mem_update_adp(dense_x, mem=mem_t,spike=spk_t,tau_m=tauM1)

        return mem_1,spk_1

    def compute_output_size(self):
        return [self.hidden_size]

# depthwide conv layer
class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, padding = 1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class SNN_DepthConv_cell(nn.Module):
    def __init__(self, input_size, output_dim, kernel_size=3,strides=1,padding=0,
                 pooling_type = None,pool_size = 2, pool_strides =2):
        super(SNN_DepthConv_cell, self).__init__()
        
        # print('SNN-conv ')
        self.input_size = input_size
        self.input_dim = input_size[0]
        self.output_dim = output_dim
    
        self.input_size = input_size
        
        
        self.rnn_name = 'SNN-conv cell'
        if pooling_type is not None: 
            if pooling_type =='max':
                self.pooling = nn.MaxPool2d(kernel_size=pool_size, stride=pool_strides, padding=1)
            elif pooling_type =='avg':
                self.pooling = nn.AvgPool2d(kernel_size=pool_size, stride=pool_strides, padding=1)
        else:
            self.pooling = None

        self.conv1_x = depthwise_separable_conv(self.input_dim,output_dim,kernel_size=kernel_size)
        self.conv_tauM = nn.Conv2d(output_dim,output_dim,kernel_size=kernel_size,stride=strides,padding=padding)
        self.sig1 = nn.Sigmoid()
        self.BN = nn.BatchNorm2d(output_dim)
        self.BN1 = nn.BatchNorm2d(output_dim)

        self.output_size = self.compute_output_size()

        nn.init.xavier_uniform_(self.conv1_x.weight)
        nn.init.xavier_uniform_(self.conv_tauM.weight)
        nn.init.constant_(self.conv1_x.bias,0)
        nn.init.constant_(self.conv_tauM.bias,0)

    def forward(self, x_t, mem_t,spk_t):

        conv_x = self.BN(self.conv1_x(x_t.float()))
        if self.pooling is not None: 
            conv_x = self.pooling(conv_x)

        tauM1 = self.sig1(self.BN1(self.conv_tauM(conv_x+mem_t)))
        # tauM1 = self.sig1(self.conv_tauM(conv_x+mem_t))

        mem_1,spk_1= mem_update_adp(conv_x, mem=mem_t,spike=spk_t,tau_m=tauM1)

        return mem_1,spk_1

    def compute_output_size(self):
        x_emp = torch.randn([1,self.input_size[0],self.input_size[1],self.input_size[2]])   
        out = self.conv1_x(x_emp)
        if self.pooling is not None: 
            out=self.pooling(out)
        return out.shape[1:]