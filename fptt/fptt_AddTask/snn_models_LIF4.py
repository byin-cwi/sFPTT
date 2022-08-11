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

def save_checkpoint(state, is_best, prefix, filename='_snn_checkpoint.pth.tar'):
    print('saving at ', prefix+filename)
    torch.save(state, prefix+filename)
    if is_best:
        shutil.copyfile(prefix+filename, prefix+ '_snn_model_best.pth.tar')


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

def mem_update_adp(inputs, mem, spike, tau_adp,tau_m, b, dt=1, isAdapt=1):
    alpha = tau_m
    
    ro = tau_adp

    # if isAdapt:
    #     beta = 1.8
    # else:
    #     beta = 0.

    # b = ro * b + (1 - ro) * spike
    # B = b_j0 + beta * b
    B=0.5


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
class sigmoid_beta(nn.Module):
    def __init__(self, alpha = 1.):
        super(sigmoid_beta,self).__init__()

        # initialize alpha
        if alpha == None:
            self.alpha = nn.Parameter(torch.tensor(1.)) # create a tensor out of alpha
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha)) # create a tensor out of alpha

            
        self.alpha.requiresGrad = False # set requiresGrad to true!
        # self.alpha=alpha

    def forward(self, x):
        if (self.alpha == 0.0):
            return x
        else:
            return torch.sigmoid(self.alpha*x)

class SNN(nn.Module):
    def __init__(self, input_size, hidden_size,output_size, n_timesteps, P=10):
        super(SNN, self).__init__()
        
        print('SNN-lc ', P)
        
        self.P = P
        self.step = n_timesteps // self.P
        
        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.output_size = output_size
        self.n_timesteps = n_timesteps
        
        self.rnn_name = 'SNN-lc cell'

        self.layer1_x = nn.Linear(input_size+hidden_size, hidden_size)
        self.layer1_tauM = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.layer1_tauAdp = nn.Linear(hidden_size+hidden_size, hidden_size)
   
        self.layer3_x = nn.Linear(hidden_size, output_size)
        self.layer3_tauM = nn.Linear(output_size+output_size, output_size)

        self.act1 = sigmoid_beta()
        self.act2 = sigmoid_beta()
        self.act3 = sigmoid_beta()

        
    def forward(self, inputs, h):
        self.fr = 0
        T = inputs.size()[0]
        
        outputs = []
        hiddens = []
 
        l,b,din = inputs.shape

        for x_i in range(T):
            x = inputs[ x_i,:,: ].view(b,1)
    
            dense_x = self.layer1_x(torch.cat((x,h[1]),dim=-1))
            tauM1 = self.act1(self.layer1_tauM(torch.cat((dense_x, h[0]),dim=-1)))
            tauAdp1 = self.act2(self.layer1_tauAdp(torch.cat((dense_x, h[2]),dim=-1)))

            mem_1,spk_1,_,b_1 = mem_update_adp(dense_x, mem=h[0],spike=h[1],
                                            tau_adp=tauAdp1,tau_m=tauM1,b = h[2])

            # dense_x = self.layer1_x(torch.cat((x,h[1]),dim=-1))
            # mem_1,spk_1,_,b_1 = mem_update_adp( dense_x, mem=h[0],spike=h[1],
            #                                 tau_adp=self.tau_adp_r1,tau_m=self.tau_m_r1,b = h[2])

            
            dense3_x = self.layer3_x(spk_1)
            tauM2 = self.act3(self.layer3_tauM(torch.cat((dense3_x, h[3]),dim=-1)))
            mem_3 = output_Neuron(dense3_x,mem=h[3],tau_m = tauM2)

            out =mem_3

            h = (mem_1,spk_1,b_1, 
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
    def __init__(self, ninp, nhid, nout, n_timesteps=784, parts=10):

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
                weight.new(bsz,self.nhid).fill_(b_j0),
                # layer 3
                weight.new(bsz,self.nout).zero_(),
                # sum spike
                weight.new(bsz,self.nout).zero_(),
                )



