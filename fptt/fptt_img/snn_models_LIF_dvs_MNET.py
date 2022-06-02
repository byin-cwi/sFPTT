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
def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
            

def model_save(fn, model, criterion, optimizer):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)

def model_load(fn):
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)
    return model, criterion, optimizer

def save_checkpoint(state, is_best, prefix, filename='_snnvgg_checkpoint.pth.tar'):
    print('saving at ', prefix+filename)
    torch.save(state, prefix+filename)
    if is_best:
        shutil.copyfile(prefix+filename, prefix+ '_snnlvgg_model_best.pth.tar')


def count_parameters(model):
    return sum(p.numel() for p in model.network.parameters() if p.requires_grad)

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
from LTC_layers import *


class SNN(nn.Module):
    def __init__(self, input_size, hidden_size,output_size, n_timesteps, P=10):
        super(SNN, self).__init__()
        
        print('SNN-ltc CNN ', P)

        self.net = ['k3c64s2', 'd-k3c64s1','k1c64s1','d-k3c128s2','k1c128s2']
        k_p = "k(.*?)c"
        c_p = "c(.*?)s"

        self.P = P
        self.step = n_timesteps // self.P
        
        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.output_size = output_size
        self.n_timesteps = n_timesteps
        
        self.rnn_name = 'SNN-mobilenet-v1'
        
        in_size = input_size
        # print(in_size)
        self.network = []
        self.network_size = []
        for i in range(len(self.net)):
            # print('in size: ',in_size)
            layer = self.net[i]
            k,c,s = int(re.search(k_p, s).group(1)),int(re.search(c_p, s).group(1)),int(s.split('s')[-1])
            if layer[0] == 'd':
                a = SNN_DepthConv_cell(in_size,c,k,s,1,)
            else:
                a = SNN_Conv_cell(in_size,c,k,s,1,)

            self.network.append(a)
            
            in_size = a.compute_output_size()
            print(in_size)
            self.network_size.append(in_size)


        self.dp_f = nn.Dropout2d(0.1)
        f_size = in_size[0]*in_size[1]*in_size[2]

        self.snn1 = SNN_rec_cell(f_size, hidden_size,is_rec=True)
        self.network_size.append([hidden_size])

        # self.snn2 = SNN_dense_cell(hidden_size, hidden_size)
   
        self.layer3_x = nn.Linear(hidden_size, output_size)
        nn.init.constant_(self.layer3_x.bias,0)
        nn.init.constant_(self.layer3_x.weight,0)
        self.network_size.append([output_size])


        self.tau_m_o = nn.Parameter(torch.Tensor(output_size))

        nn.init.constant_(self.tau_m_o, 5.)

        self.act3 = nn.Sigmoid()

        
    def forward(self, inputs, h):
        self.fr = 0
        # T = inputs.size()[0]
        
        # outputs = []
        hiddens = []
        h = list(h)
 
        b,c,w,r = inputs.shape

        x_in = inputs.view(b,c,w,r)

        for layer_i in range(len(self.network)):
            layer_ = self.network[layer_i]
            h[2*layer_i],h[1+2*layer_i] = layer_(x_in,h[2*layer_i],h[1+2*layer_i])
            x_in = h[1+2*layer_i]
            self.fr = self.fr+ x_in.detach().cpu().numpy().mean()


        # spk_conv = self.dp_f(x_in)
        # f_spike = torch.flatten(spk_conv,1)
        f_spike = torch.flatten(x_in,1)


        layer_i +=1
        h[2*layer_i],h[1+2*layer_i]= self.snn1.forward(f_spike,h[2*layer_i],h[1+2*layer_i])
        self.fr = self.fr+ h[1+2*layer_i].detach().cpu().numpy().mean()

        
        dense3_x = self.layer3_x(h[1+2*layer_i])
        tauM2 = self.act3(self.tau_m_o)
        h[-2] = output_Neuron(dense3_x,mem=h[-2],tau_m = tauM2)

        h[-1] =h[-2]
        h = tuple(h)

        f_output = F.log_softmax(h[-1], dim=1)
        # outputs.append(f_output)
        hiddens.append(h)

        self.fr = self.fr/(len(self.network)+1.)
                
        final_state = h

        return f_output, final_state, hiddens

class SeqModel(nn.Module):
    def __init__(self, ninp, nhid, nout, dropout=0.0, dropouti=0.0, dropouth=0.0, wdrop=0.0,
                 temporalwdrop=False, wnorm=True, n_timesteps=784, nfc=256, parts=10):

        super(SeqModel, self).__init__()
        self.nout = nout    # Should be the number of classes
        self.nhid = nhid

        self.rnn_name = 'SNN vgg'

        self.network = SNN(input_size=ninp, hidden_size=nhid, output_size=nout,n_timesteps=n_timesteps, P=parts)
        self.layer_size = self.network.network_size
        print(self.layer_size)
        
        self.l2_loss = nn.MSELoss()

    def forward(self, inputs, hidden):
        # inputs = inputs.permute(2, 0, 1)  
        # print(inputs.shape) # L,B,d
        b,l,c,w,h = inputs.shape
        outputs = []
        for i in range(l):
            f_output, hidden, hiddens= self.network.forward(inputs[:,i,:,:,:], hidden)
            outputs.append(f_output)
        recon_loss = torch.zeros(1, device=inputs.device)
        return outputs, hidden, recon_loss

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        states = []
        for l in self.layer_size:
            if len(l) == 3:
                states.append(weight.new(bsz,l[0],l[1],l[2]).uniform_())
                states.append(weight.new(bsz,l[0],l[1],l[2]).zero_())
            elif len(l) == 1:
                states.append(weight.new(bsz,l[0]).uniform_())
                states.append(weight.new(bsz,l[0]).zero_())

        states.append(weight.new(bsz,self.nout).zero_())
        states.append(weight.new(bsz,self.nout).zero_())
        # print(self.layer_size)
        # print([s.shape for s in states])
        return tuple(states)



