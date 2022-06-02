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

def save_checkpoint(state, is_best, prefix, filename='_snnl4s3_checkpoint.pth.tar'):
    print('saving at ', prefix+filename)
    torch.save(state, prefix+filename)
    if is_best:
        shutil.copyfile(prefix+filename, prefix+ '_snnl4s3_model_best.pth.tar')


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

    if isAdapt:
        beta = 1.8
    else:
        beta = 0.

    b = ro * b + (1 - ro) * spike
    B = b_j0 + beta * b
    # B = .2



    d_mem = -mem + inputs
    mem = mem + d_mem*alpha
    inputs_ = mem - B

    spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
    mem = (1-spike)*mem

    return mem, spike, b


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
        self.layer1_tauM = nn.Linear(2*hidden_size, hidden_size)
        self.layer1_tauAdp = nn.Linear(2*hidden_size, hidden_size)

        self.act1 = nn.Sigmoid()

        nn.init.xavier_uniform_(self.layer1_x.weight)
        nn.init.xavier_uniform_(self.layer1_tauM.weight)

    def forward(self, x_t, mem_t,spk_t,b_t):    
        if self.is_rec:
            dense_x = self.layer1_x(torch.cat((x_t,spk_t),dim=-1))
        else:
            dense_x = self.layer1_x(x_t)

        tauM1 = self.act1(self.layer1_tauM(torch.cat((dense_x,mem_t),dim=-1)))
        tauAdp1 = self.act1(self.layer1_tauAdp(torch.cat((dense_x,b_t),dim=-1)))
        

        mem_1,spk_1,b_1 = mem_update_adp(dense_x, mem=mem_t,spike=spk_t,b=b_t,
                                        tau_m=tauM1, tau_adp=tauAdp1)

        return mem_1,spk_1,b_1

    def compute_output_size(self):
        return [self.hidden_size]


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
                self.pooling = nn.MaxPool2d(kernel_size=pool_size, stride=pool_strides, padding=0)
            elif pooling_type =='avg':
                self.pooling = nn.AvgPool2d(kernel_size=pool_size, stride=pool_strides, padding=0)
        else:
            self.pooling = None

        self.conv_x = nn.Conv2d(self.input_dim,output_dim,kernel_size=kernel_size,stride=strides,padding=padding)
        self.conv_tauM = nn.Conv2d(output_dim,output_dim,kernel_size=kernel_size,stride=strides,padding=padding)
        self.conv_tauAdp = nn.Conv2d(output_dim,output_dim,kernel_size=kernel_size,stride=strides,padding=padding)
        self.sig1 = nn.Sigmoid()
        self.sig2 = nn.Sigmoid()
        self.sig3 = nn.Sigmoid()
        self.BN1 = nn.BatchNorm2d(output_dim)
        self.BN2 = nn.BatchNorm2d(output_dim)
        self.BN3 = nn.BatchNorm2d(output_dim)

        # nn.init.kaiming_uniform_(self.conv_x.weight)
        # nn.init.kaiming_uniform_(self.conv_tauM.weight)
        # nn.init.kaiming_uniform_(self.conv_tauAdp.weight)
        nn.init.xavier_uniform_(self.conv_x.weight)
        nn.init.xavier_uniform_(self.conv_tauM.weight)
        nn.init.xavier_uniform_(self.conv_tauAdp.weight)
        nn.init.constant_(self.conv_x.bias,0)
        nn.init.constant_(self.conv_tauM.bias,0)
        nn.init.constant_(self.conv_tauAdp.bias,0)

    def forward(self, x_t, mem_t,spk_t,b_t):
        conv_x = self.BN1(self.conv_x(x_t.float()))
        
        if self.pooling is not None: 
            conv_x = self.pooling(conv_x)
      
        # tauM1 = self.sig2(self.conv_tauM(torch.cat((conv_x, mem_t),dim=1)))
        tauM1 = self.sig2(self.BN2(self.conv_tauM(conv_x+mem_t)))
        # tauAdp1 = self.sig3(self.conv_tauAdp(torch.cat((conv_x,b_t),dim=1)))
        tauAdp1 = self.sig3(self.BN3(self.conv_tauAdp(conv_x+b_t)))

        

        mem_1,spk_1,b_1 = mem_update_adp(conv_x, mem=mem_t,spike=spk_t,
                                        tau_adp=tauAdp1,tau_m=tauM1,b =b_t)

        return mem_1,spk_1,b_1

    def compute_output_size(self):
        x_emp = torch.randn([1,self.input_size[0],self.input_size[1],self.input_size[2]])   
        out = self.conv_x(x_emp)
        if self.pooling is not None: out=self.pooling(out)
        return out.shape[1:]

class SNN(nn.Module):
    def __init__(self, input_size, hidden_size,output_size, n_timesteps, P=10):
        super(SNN, self).__init__()
        
        print('SNN-ltc CNN ', P)
        
        self.P = P
        self.step = n_timesteps // self.P
        
        self.input_size = [1,28,28]#input_size
        self.hidden_size = hidden_size 
        self.output_size = output_size
        self.n_timesteps = n_timesteps
        
        self.rnn_name = 'SNN-ltc'
        self.conv1 = SNN_Conv_cell(self.input_size,32,3,1,1,None)
        self.conv2 = SNN_Conv_cell(self.conv1.compute_output_size(),128,3,1,1,'max')
        self.conv3 = SNN_Conv_cell(self.conv2.compute_output_size(),256,3,1,1,'max')

        f_shape = self.conv3.compute_output_size()
        self.snn1 = SNN_rec_cell(f_shape[0]*f_shape[1]*f_shape[2] , hidden_size,is_rec=False)
        self.snn2 = SNN_rec_cell(hidden_size, hidden_size,False)
   
        self.layer3_x = nn.Linear(hidden_size, output_size)
        self.layer3_tauM = nn.Linear(output_size+output_size, output_size)
        self.tau_m_o = nn.Parameter(torch.Tensor(output_size))

        nn.init.constant_(self.tau_m_o, 3.)
        self.dp_in = nn.Dropout(0.1)

        self.act3 = sigmoid_beta()
        self.layers = [self.conv1,self.conv2,self.conv3,self.snn1,self.snn2]

    def list_of_sizes(self):
        return tuple([a.compute_output_size() for a in self.layers])

        
    def forward(self, inputs, h):
        self.fr = 0
        # T = inputs.size()[0]
        
        # outputs = []
        hiddens = []
     
        # print(inputs.shape)
        b,_,_ = inputs.shape
        

        # for x_i in range(T):
        x = inputs.view(b,1,28,28)
        # x = self.dp_in(x)

        layer_i = 0
        mem_conv1,spk_conv1,b_conv1 = self.conv1(x,h[layer_i*3],h[1+layer_i*3],h[2+layer_i*3])
        layer_i +=1
        mem_conv2,spk_conv2,b_conv2 = self.conv2(spk_conv1,h[layer_i*3],h[1+layer_i*3],h[2+layer_i*3])
        layer_i +=1
        mem_conv3,spk_conv3,b_conv3 = self.conv3(spk_conv2,h[layer_i*3],h[1+layer_i*3],h[2+layer_i*3])

        f_spike = torch.flatten(spk_conv3,1)
        # print(f_spike.shape)
        layer_i +=1
        mem_1,spk_1,b_1 = self.snn1.forward(f_spike,h[layer_i*3],h[1+layer_i*3],h[2+layer_i*3])
        layer_i +=1
        mem_2,spk_2,b_2 = self.snn2.forward(spk_1,h[layer_i*3],h[1+layer_i*3],h[2+layer_i*3])

        
        dense3_x = self.layer3_x(spk_2)
        tauM2 = self.act3(self.tau_m_o)#self.act3(self.layer3_tauM(torch.cat((dense3_x, h[3]),dim=-1)))
        tauM2 = self.act3(self.layer3_tauM(torch.cat((dense3_x, h[-2]),dim=-1)))
        mem_o = output_Neuron(dense3_x,mem=h[-2],tau_m = tauM2)

        out =mem_o

        h = (mem_conv1,spk_conv1,b_conv1,
            mem_conv2,spk_conv2,b_conv2,
            mem_conv3,spk_conv3,b_conv3,
   
            mem_1,spk_1,b_1, 
            mem_2,spk_2,b_2, 
            mem_o,
            out)

        f_output = F.log_softmax(out, dim=1)
        # outputs.append(f_output)
        hiddens.append(h)
        
        for spk in [spk_conv1,spk_conv2,spk_1,spk_2]:
            self.fr = self.fr+ spk.detach().cpu().numpy().mean()/5.
                
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

        self.rnn_name = 'SNN FF'

        self.network = SNN(input_size=ninp, hidden_size=nhid, output_size=nout,n_timesteps=n_timesteps, P=parts)
        self.layer_size = self.network.list_of_sizes()
        
        self.l2_loss = nn.MSELoss()

    def forward(self, inputs, hidden):
        # inputs = inputs.permute(2, 0, 1)  
        # print(inputs.shape) # L,B,d
        outputs = []
        for i in range(20):
            f_output, hidden, hiddens= self.network.forward(inputs, hidden)
            outputs.append(f_output)
        recon_loss = torch.zeros(1, device=inputs.device)
        return outputs, hidden, recon_loss

    def init_hidden(self, bsz):
        states = []
        weight = next(self.parameters()).data
        for l in self.layer_size:
            if len(l) == 3:

                states.append(weight.new(bsz,l[0],l[1],l[2]).uniform_())
                states.append(weight.new(bsz,l[0],l[1],l[2]).zero_())
                states.append(weight.new(bsz,l[0],l[1],l[2]).zero_())
            elif len(l) == 1:

                states.append(weight.new(bsz,l[0]).uniform_())
                states.append(weight.new(bsz,l[0]).zero_())
                states.append(weight.new(bsz,l[0]).zero_())


        states.append(weight.new(bsz,self.nout).zero_())
        states.append(weight.new(bsz,self.nout).zero_())
        # print(self.layer_size)
        return tuple(states)



