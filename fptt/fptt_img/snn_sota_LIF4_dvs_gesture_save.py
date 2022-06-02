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

def save_checkpoint(state, is_best, prefix, filename='_snnl4s2_checkpoint.pth.tar'):
    print('saving at ', prefix+filename)
    torch.save(state, prefix+filename)
    if is_best:
        shutil.copyfile(prefix+filename, prefix+ '_snnl4s2_model_best.pth.tar')


def count_parameters(model):
    return sum(p.numel() for p in model.network.parameters() if p.requires_grad)

###############################################################################################
###############################    Define SNN layer   #########################################
###############################################################################################

b_j0 = .5  # neural threshold baseline
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
    
    # ro = tau_adp

    # if isAdapt:
    #     beta = 1.8
    # else:
    #     beta = 0.

    # b = ro * b + (1 - ro) * spike
    # B = b_j0 + beta * b
    B = 1.


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

class SNN_Conv_cell(nn.Module):
    def __init__(self, input_size, output_dim, kernel_size=5,strides=1,padding=0,
                 pooling_type = None,pool_size = 2, pool_strides =2):
        super(SNN_Conv_cell, self).__init__()
        
        print('SNN-conv ')
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
        # self.conv2_x = nn.Conv2d(output_dim,output_dim,kernel_size=kernel_size,stride=strides,padding=padding)
        # self.conv3_x = nn.Conv2d(output_dim,output_dim,kernel_size=kernel_size,stride=strides,padding=padding)
        self.conv_tauM = nn.Conv2d(output_dim,output_dim,kernel_size=3,stride=strides,padding=1)
        self.conv_tauAdp = nn.Conv2d(output_dim,output_dim,kernel_size=3,stride=strides,padding=1)
        self.sig1 = nn.Sigmoid()
        self.sig2 = nn.Sigmoid()
        self.sig3 = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.BN = nn.BatchNorm2d(output_dim)
        self.BN1 = nn.BatchNorm2d(output_dim)
        self.BN2 = nn.BatchNorm2d(output_dim)

        self.output_size = self.compute_output_size()

        self.tau_adp = nn.Parameter(torch.Tensor(self.output_size))
        self.tau_m = nn.Parameter(torch.Tensor(self.output_size))

        nn.init.xavier_uniform_(self.conv1_x.weight)
        # nn.init.xavier_uniform_(self.conv2_x.weight)
        # nn.init.xavier_uniform_(self.conv3_x.weight)
        nn.init.xavier_uniform_(self.conv_tauM.weight)
        nn.init.xavier_uniform_(self.conv_tauAdp.weight)

        nn.init.normal_(self.tau_adp, 4, 1)
        nn.init.normal_(self.tau_adp, 0.,1)


    def forward(self, x_t, mem_t,spk_t,b_t):
        # conv1_x = self.relu(self.BN1(self.conv1_x(x_t.float())))
        # conv2_x = self.relu(self.BN2(self.conv2_x(conv1_x)))
        # conv_x = self.BN(self.conv3_x(conv2_x))
        conv_x = self.BN(self.conv1_x(x_t.float()))
        if self.pooling is not None: 
            conv_x = self.pooling(conv_x)
            # tauM1 = self.pooling(tauM1)
            # tauAdp1 = self.pooling(tauAdp1)
        # print(conv_x.shape,mem_t.shape)
        # print(torch.cat((conv_x, mem_t),dim=1).shape)
        # tauM1 = self.sig2(self.conv_tauM(torch.cat((conv_x, mem_t),dim=1)))
        # tauAdp1 = self.sig3(self.conv_tauAdp(torch.cat((conv_x,b_t),dim=1)))
        tauM1 = self.sig2(self.BN1(self.conv_tauM(conv_x+mem_t)))
        tauAdp1 = self.sig3(self.conv_tauAdp(conv_x+b_t))

        # tauM1 = self.sig2(self.tau_adp)
        # tauAdp1 = self.sig3(self.tau_m)

        mem_1,spk_1,_,b_1 = mem_update_adp(conv_x, mem=mem_t,spike=spk_t,
                                        tau_adp=tauAdp1,tau_m=tauM1,b =b_t)

        return mem_1,spk_1,b_1

    def compute_output_size(self):
        x_emp = torch.randn([1,self.input_size[0],self.input_size[1],self.input_size[2]])   
        # out = self.conv2_x(self.conv1_x(x_emp))
        out = self.conv1_x(x_emp)
        if self.pooling is not None: 
            out=self.pooling(out)
        return out.shape[1:]

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

        # self.tau_adp = nn.Parameter(torch.Tensor(hidden_size))
        # self.tau_m =nn.Parameter(torch.Tensor(hidden_size))
        self.act1 = nn.Sigmoid()

        # nn.init.normal_(self.tau_adp, 4.6,.1)
        # nn.init.normal_(self.tau_m, 3.,.1)
        nn.init.xavier_uniform_(self.layer1_x.weight)
        nn.init.xavier_uniform_(self.layer1_tauAdp.weight)
        nn.init.xavier_uniform_(self.layer1_tauM.weight)

    def forward(self, x_t, mem_t,spk_t,b_t):    
        if self.is_rec:
            dense_x = self.layer1_x(torch.cat((x_t,spk_t),dim=-1))
        else:
            dense_x = self.layer1_x(x_t)
        # tauM1 = self.act1(self.tau_m)
        # tauAdp1 = self.act2(self.tau_adp)
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
        
        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.output_size = output_size
        self.n_timesteps = n_timesteps
        
        self.rnn_name = 'SNN-lc cell'
        self.conv_snn1 = SNN_Conv_cell([2,128,128],64,7,1,1,'max',pool_size=4,pool_strides=4)
        self.conv_snn2 = SNN_Conv_cell([64,31,31],128,3,1,1,'max')
        self.conv_snn3 = SNN_Conv_cell([128,16,16],128,3,1,1,'max')
        self.conv_snn4 = SNN_Conv_cell([128,9,9],256,3,1,1,'max')

        self.snn_rec = SNN_rec_cell(256*25,hidden_size)

        print(self.conv_snn1.compute_output_size(),
                self.conv_snn2.compute_output_size(),
                self.conv_snn3.compute_output_size(),self.conv_snn4.compute_output_size())
        

        self.layer3_x = nn.Linear(hidden_size,output_size)
        # self.layer3_tauM = nn.Linear(output_size*2,output_size)
        self.tau_m_o = nn.Parameter(torch.Tensor(output_size))

        nn.init.constant_(self.tau_m_o, 3.)#4.
        # nn.init.constant_(self.tau_m_o, 0.)
        # nn.init.xavier_uniform_(self.layer3_x.weight)

        # self.act1 = nn.Sigmoid()
        # self.act2 = nn.Sigmoid()
        self.act3 = nn.Sigmoid()

        
    def forward(self, inputs, h):
        self.fr = 0
        T = inputs.size()[1]
        
        # outputs = []
        hiddens = []
 
        b,c,d1,d2 = inputs.shape

        # for x_i in range(T):
        # x_down = F.avg_pool2d(inputs[ :,:,:,: ],kernel_size=4,stride=4)

        mem_1,spk_1,b_1 = self.conv_snn1(inputs, mem_t=h[0],spk_t=h[1],b_t = h[2])

        mem_2,spk_2,b_2 = self.conv_snn2(spk_1, mem_t=h[3],spk_t=h[4],b_t = h[5])

        mem_3,spk_3,b_3 = self.conv_snn3(spk_2, mem_t=h[6],spk_t=h[7],b_t = h[8])

        mem_4,spk_4,b_4 = self.conv_snn4(spk_3, mem_t=h[9],spk_t=h[10],b_t = h[11])
        f_spike = torch.flatten(spk_4,1)
        
        # mem_3,spk_3,b_3 = self.snn3(torch.cat((spk_2_dp,spk_1_dp),dim=-1), mem_t=h[6],spk_t=h[7],b_t = h[8])
        mem_5,spk_5,b_5 = self.snn_rec(f_spike, mem_t=h[12],spk_t=h[13],b_t = h[14])
        # spk_3_dp = self.dp3(spk_3)

        dense3_x = self.layer3_x(spk_5)
        tauM2 = self.act3(self.tau_m_o)
        # tauM2 = self.act3(self.layer3_tauM(torch.cat((dense3_x, h[-2]),dim=-1)))
        mem_out = output_Neuron(dense3_x,mem=h[-2],tau_m = tauM2)

        out =mem_out

        h = (mem_1,spk_1,b_1, 
            mem_2,spk_2,b_2, 
            mem_3,spk_3,b_3, 
            mem_4,spk_4,b_4, 
            mem_5,spk_5,b_5, 
            mem_out,
            out)

        f_output = F.log_softmax(out, dim=1)
        # outputs.append(f_output)
        hiddens.append(h)

        self.fr = self.fr+ spk_1.detach().cpu().numpy().mean()/3\
                + spk_2.detach().cpu().numpy().mean()/3\
                + spk_3.detach().cpu().numpy().mean()/3
                
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
        return (weight.new(bsz,64,31,31).uniform_(),
                weight.new(bsz,64,31,31).zero_(),
                weight.new(bsz,64,31,31).fill_(b_j0),
                # layer 2
                weight.new(bsz,128,16,16).uniform_(),
                weight.new(bsz,128,16,16).zero_(),
                weight.new(bsz,128,16,16).fill_(b_j0),
                # layer 3
                weight.new(bsz,128,9,9).uniform_(),
                weight.new(bsz,128,9,9).zero_(),
                weight.new(bsz,128,9,9).fill_(b_j0),
                # layer 3
                weight.new(bsz,128*2,5,5).uniform_(),
                weight.new(bsz,128*2,5,5).zero_(),
                weight.new(bsz,128*2,5,5).fill_(b_j0),
                # layer out
                weight.new(bsz,self.nhid).uniform_(),
                weight.new(bsz,self.nhid).zero_(),
                weight.new(bsz,self.nhid).fill_(b_j0),
                # sum spike
                weight.new(bsz,self.nout).zero_(),
                weight.new(bsz,self.nout).zero_(),
                )



