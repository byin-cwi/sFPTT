"""
Att gated snn
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

def save_checkpoint(state, is_best, prefix, filename='_att-snn_checkpoint.pth.tar'):
    print('saving at ', prefix+filename)
    torch.save(state, prefix+filename)
    if is_best:
        shutil.copyfile(prefix+filename, prefix+ '_att-snn_model_best.pth.tar')


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

def output_Neuron(inputs, mem, tau_m, dt=1):
    """
    The read out neuron is leaky integrator without spike
    """
    d_mem = -mem  +  inputs
    mem = mem+d_mem
    return mem
###############################################################################################
###############################################################################################
###############################################################################################
import torch
from torch import nn

class LSTM_cell(torch.nn.Module):
    def __init__(self, input_dim=10, hidden_dim=20):
        super(LSTM_cell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # forget gate components
        self.linear_forget_w1 = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.linear_forget_r1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.sigmoid_forget = nn.Sigmoid()

        # input gate components
        self.linear_gate_w2 = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.linear_gate_r2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.sigmoid_gate = nn.Sigmoid()

        # cell memory components
        self.linear_gate_w3 = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.linear_gate_r3 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activation_gate = nn.Tanh()

        # out gate components
        self.linear_gate_w4 = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.linear_gate_r4 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.sigmoid_hidden_out = nn.Sigmoid()

        self.activation_final = nn.Tanh()

    def forget(self, x, h):
        x = self.linear_forget_w1(x)
        h = self.linear_forget_r1(h)
        return self.sigmoid_forget(x + h)

    def input_gate(self, x, h):
        # Equation 1. input gate
        x_temp = self.linear_gate_w2(x)
        h_temp = self.linear_gate_r2(h)
        i = self.sigmoid_gate(x_temp + h_temp)
        return i

    def cell_memory_gate(self, i, f, x, h, c_prev):
        x = self.linear_gate_w3(x)
        h = self.linear_gate_r3(h)

        # new information part that will be injected in the new context
        k = self.activation_gate(x + h)
        g = k * i

        # forget old context/cell info
        c = f * c_prev
        # learn new context/cell info
        c_next = g + c
        return c_next

    def out_gate(self, x, h):
        x = self.linear_gate_w4(x)
        h = self.linear_gate_r4(h)
        return self.sigmoid_hidden_out(x + h)

    def forward(self, x, mem_t,c_t):
        i = self.input_gate(x, mem_t)

        f = self.forget(x, mem_t)

        c_next = self.cell_memory_gate(i, f, x, mem_t,c_t)

        o = self.out_gate(x, mem_t)

        mem_next = o * self.activation_final(c_next)
        # generate spk 
        spk = act_fun_adp(mem_next-0.1)
        # reset mem
        mem_reset = (1-spk)*mem_next

        return mem_reset, c_next, spk

class SNN(nn.Module):
    def __init__(self, input_size, hidden_size,output_size, n_timesteps, P=10):
        super(SNN, self).__init__()
        
        print('SNN-lstm-type2 ', P)
        
        self.P = P
        self.step = n_timesteps // self.P
        
        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.output_size = output_size
        self.n_timesteps = n_timesteps
        
        self.rnn_name = 'SNN-lstm type-2 cell'


        self.lstm_snn = LSTM_cell(input_size,hidden_size)


        self.layer3_x = nn.Linear(hidden_size, output_size)
        self.layer3_tauM = nn.Linear(output_size*2, output_size)
        self.tau_m_o = nn.Parameter(torch.Tensor(output_size))
        nn.init.constant_(self.tau_m_o, 3.)
        
    def forward(self, inputs, h):
        self.fr = 0
        T = inputs.size()[0]
        
        outputs = []
        hiddens = []
 
        l,b,din = inputs.shape

        for x_i in range(T):
            x = inputs[ x_i,:,: ].view(b,1)

            mem_1,c_1,spk_1 = self.lstm_snn(x,h[0],h[1])
            
            dense3_x = self.layer3_x(spk_1)
            tauM2 = torch.sigmoid(self.tau_m_o)#torch.sigmoid(self.layer3_tauM(torch.cat((dense3_x, h[2]),dim=-1)))
            mem_3 = output_Neuron(dense3_x,mem=h[2],tau_m = tauM2)

            out = mem_3

            h = (mem_1,c_1,
                mem_3,
                out)

            f_output = F.log_softmax(out, dim=1)
            outputs.append(f_output)
            hiddens.append(h)

            self.fr = self.fr+ spk_1.detach().cpu().numpy().mean()
                
        # output = torch.as_tensor(outputs)
        final_state = h
        self.fr = self.fr/T
        return outputs, final_state, hiddens

class SeqModel(nn.Module):
    def __init__(self, ninp, nhid, nout, dropout=0.0, dropouti=0.0, dropouth=0.0, wdrop=0.0,
                 temporalwdrop=False, wnorm=True, n_timesteps=784, nfc=256, parts=10):

        super(SeqModel, self).__init__()
        self.nout = nout    # Should be the number of classes
        self.nhid = nhid

        self.rnn_name = 'SNN'

        self.network = SNN(input_size=ninp, hidden_size=nhid, output_size=nout,n_timesteps=n_timesteps, P=parts)
        
        self.l2_loss = nn.MSELoss()

    def forward(self, inputs, hidden):
        inputs = inputs.permute(2, 0, 1)  
        # print(inputs.shape) # L,B,d

        outputs, hidden, hiddens= self.network.forward(inputs, hidden)

        recon_loss = torch.zeros(1, device=inputs.device)
        return outputs, hidden, recon_loss

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new(bsz,self.nhid).uniform_(),
                weight.new(bsz,self.nhid).zero_(),
                # layer 3
                weight.new(bsz,self.nout).zero_(),
                # sum spike
                weight.new(bsz,self.nout).zero_(),
                )



