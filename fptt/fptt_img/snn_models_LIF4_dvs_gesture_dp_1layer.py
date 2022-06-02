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

def save_checkpoint(state, is_best, prefix, filename='_snnl4s2-max_checkpoint.pth.tar'):
    print('saving at ', prefix+filename)
    torch.save(state, prefix+filename)
    if is_best:
        shutil.copyfile(prefix+filename, prefix+ '_snnl4s2-max_model_best.pth.tar')


def count_parameters(model):
    return sum(p.numel() for p in model.network.parameters() if p.requires_grad)

###############################################################################################
###############################    Define SNN layer   #########################################
###############################################################################################

b_j0 = .1  # neural threshold baseline, default 0.1
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

def mem_update_adp(inputs, mem, spike, tau_adp,tau_m, b, dt=1, isAdapt=1):
    alpha = tau_m
    
    ro = tau_adp

    if isAdapt:
        beta = 1.8
    else:
        beta = 0.

    b = ro * b + (1 - ro) * spike
    B = b_j0 + beta * b
    # B = 1.


    d_mem = -mem + inputs
    mem = mem + d_mem*alpha
    inputs_ = mem - B

    spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
    mem = (1-spike)*mem

    return mem, spike, B, b


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
class SNN_rec_cell(nn.Module):
    def __init__(self, input_size, hidden_size,is_rec = True):
        super(SNN_rec_cell, self).__init__()
        # print('SNN-ltc ')
    
        
        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.is_rec = is_rec
        # self.rnn_name = 'SNN-ltc cell'

        if is_rec:
            self.layer1_x = nn.Linear(input_size+hidden_size, hidden_size)
        else:
            self.layer1_x = nn.Linear(input_size, hidden_size)
        self.layer1_tauAdp = nn.Linear(2*hidden_size, hidden_size)
        self.layer1_tauM = nn.Linear(2*hidden_size, hidden_size)

        self.act1 = nn.Sigmoid()


        nn.init.xavier_uniform_(self.layer1_x.weight)
        nn.init.xavier_uniform_(self.layer1_tauAdp.weight)
        nn.init.xavier_uniform_(self.layer1_tauM.weight)

    def forward(self, x_t, mem_t,spk_t,b_t):    
        if self.is_rec:
            dense_x = self.layer1_x(torch.cat((x_t,spk_t),dim=-1))
        else:
            dense_x = self.layer1_x(x_t)

        tauM1 = self.act1(self.layer1_tauM(torch.cat((dense_x,mem_t),dim=-1)))
        tauAdp1 = self.act1(self.layer1_tauAdp(torch.cat((dense_x,b_t),dim=-1)))
        

        mem_1,spk_1,_,b_1 = mem_update_adp(dense_x, mem=mem_t,spike=spk_t,
                                        tau_adp=tauAdp1,tau_m=tauM1,b =b_t)

        return mem_1,spk_1,b_1

    def compute_output_size(self):
        return [self.hidden_size]

class SNN(nn.Module):
    def __init__(self, input_size, hidden_size,output_size, n_timesteps, P=10):
        super(SNN, self).__init__()
        
        print('SNN-lc ', P)
        
        self.P = P
        self.step = n_timesteps // self.P
        input_size = 32*32
        self.input_size =input_size
        self.hidden_size = hidden_size 
        self.output_size = output_size
        self.n_timesteps = n_timesteps

        # self.snn1 = SNN_rec_cell(input_size,hidden_size,False)
        # self.snn2 = SNN_rec_cell(input_size,hidden_size,False)
        self.snn3 = SNN_rec_cell(2048,hidden_size)
        

        self.layer3_x = nn.Linear(hidden_size,output_size)
        # self.layer3_tauM = nn.Linear(output_size*2,output_size)
        self.tau_m_o = nn.Parameter(torch.Tensor(output_size))

        nn.init.constant_(self.tau_m_o, 0.)
        nn.init.xavier_uniform_(self.layer3_x.weight)

        self.act1 = nn.Sigmoid()
        self.act2 = nn.Sigmoid()
        self.act3 = nn.Sigmoid()

        self.dp1 = nn.Dropout(0.1)#.1
        self.dp2 = nn.Dropout(0.1)
        self.dp3 = nn.Dropout(0.1)

        
    def forward(self, inputs, h):
        self.fr = 0
        T = inputs.size()[1]
        
        # outputs = []
        hiddens = []
 
        b,c,d1,d2 = inputs.shape

        # for x_i in range(T):
        x_down = F.avg_pool2d(inputs[ :,:,:,: ],kernel_size=4,stride=4).view(b,2048)
        # x_down = F.max_pool2d(inputs[ :,:,:,: ],kernel_size=4,stride=4) # 1000
        # x1 = x_down[:,0,:,:].view(b,self.input_size)
        # x2 = x_down[:,1,:,:].view(b,self.input_size)

        # mem_1,spk_1,b_1 = self.snn1(x_down[:,0,:,:].view(b,self.input_size), mem_t=h[0],spk_t=h[1],b_t = h[2])

        # mem_2,spk_2,b_2 = self.snn2(x_down[:,1,:,:].view(b,self.input_size), mem_t=h[3],spk_t=h[4],b_t = h[5])

        # spk_1_dp = self.dp1(spk_1)
        # spk_2_dp = self.dp2(spk_2)
        
        mem_3,spk_3,b_3 = self.snn3(x_down, mem_t=h[0],spk_t=h[1],b_t = h[2])
        spk_3_dp = self.dp3(spk_3)

        dense3_x = self.layer3_x(spk_3_dp)
        tauM2 = self.act3(self.tau_m_o)
        # tauM2 = self.act3(self.layer3_tauM(torch.cat((dense3_x, h[-2]),dim=-1)))
        mem_out = output_Neuron(dense3_x,mem=h[-1],tau_m = tauM2)

        out =mem_out

        h = (
            mem_3,spk_3,b_3, 
            mem_out)

        f_output = F.log_softmax(out, dim=1)
        # outputs.append(f_output)
        hiddens.append(h)

        self.fr = self.fr+ spk_3.detach().cpu().numpy().mean()
                
        # output = torch.as_tensor(outputs)
        final_state = h
        # self.fr = self.fr/T
        return f_output, final_state, hiddens

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
        # inputs = inputs.permute(2, 0, 1)  
        t = inputs.size()[1]
        # print(inputs.shape) # L,B,d

        outputs = []
        for i in range(t):
            f_output, hidden, hiddens= self.network.forward(inputs[:,i,:,:,:], hidden)
            outputs.append(f_output)
        recon_loss = torch.zeros(1, device=inputs.device)
        return outputs, hidden, recon_loss

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (
                weight.new(bsz,self.nhid).uniform_(),
                weight.new(bsz,self.nhid).zero_(),
                weight.new(bsz,self.nhid).fill_(b_j0),
                # layer out
                weight.new(bsz,self.nout).zero_(),
                # # sum spike
                # weight.new(bsz,self.nout).zero_(),
                )



