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

def save_checkpoint(state, is_best, prefix, filename='_snn_sota_2layer_checkpoint.pth.tar'):
    print('saving at ', prefix+filename)
    torch.save(state, prefix+filename)
    if is_best:
        shutil.copyfile(prefix+filename, prefix+ '_snn_model_sota_2layer_best.pth.tar')


def count_parameters(model):
    return sum(p.numel() for p in model.network.parameters() if p.requires_grad)

class SeparatedBatchNorm1d(nn.Module):

    """
    A batch normalization module which keeps its running mean
    and variance separately per timestep.
    """

    def __init__(self, num_features, max_length, eps=1e-5, momentum=0.1,
                 affine=True):
        """
        Most parts are copied from
        torch.nn.modules.batchnorm._BatchNorm.
        """

        super(SeparatedBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.max_length = max_length
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if self.affine:
            self.weight = nn.Parameter(torch.FloatTensor(num_features))
            self.bias = nn.Parameter(torch.FloatTensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        for i in range(max_length):
            self.register_buffer(
                'running_mean_{}'.format(i), torch.zeros(num_features))
            self.register_buffer(
                'running_var_{}'.format(i), torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.max_length):
            running_mean_i = getattr(self, 'running_mean_{}'.format(i))
            running_var_i = getattr(self, 'running_var_{}'.format(i))
            running_mean_i.zero_()
            running_var_i.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input_):
        if input_.size(1) != self.running_mean_0.nelement():
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input_.size(1), self.num_features))

    def forward(self, input_, time):
        self._check_input_dim(input_)
        if time >= self.max_length:
            time = self.max_length - 1
        running_mean = getattr(self, 'running_mean_{}'.format(time))
        running_var = getattr(self, 'running_var_{}'.format(time))
        return F.batch_norm(
            input=input_, running_mean=running_mean, running_var=running_var,
            weight=self.weight, bias=self.bias, training=self.training,
            momentum=self.momentum, eps=self.eps)

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' max_length={max_length}, affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))
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

    d_mem = -mem + inputs
    mem = mem + d_mem*alpha
    # mem = (1-alpha)*(1-spike)*mem+inputs
    inputs_ = mem - B

    spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
    mem = (1-spike)*mem

    return mem, spike, B, b


def output_Neuron(inputs, mem, tau_m, dt=1):
    """
    The read out neuron is leaky integrator without spike
    """
    d_mem = inputs#-mem  +  inputs
    # mem = mem+d_mem*tau_m
    mem = (1-tau_m)*mem+d_mem*tau_m
  
    return mem
###############################################################################################
###############################################################################################
###############################################################################################
class sigmoid_beta(nn.Module):
    def __init__(self, alpha = 1.,is_train=False):
        super(sigmoid_beta,self).__init__()

        # initialize alpha
        if alpha == None:
            self.alpha = nn.Parameter(torch.tensor(1.)) # create a tensor out of alpha
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha)) # create a tensor out of alpha

            
        self.alpha.requiresGrad = is_train # set requiresGrad to true!
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
        
        self.rnn_name = 'SNN-lc 2 layer cell'

        self.layer1_x = nn.Linear(input_size, hidden_size)
        self.layer1_r = nn.Linear(hidden_size, hidden_size)
        self.layer1_tauM = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.layer1_tauAdp = nn.Linear(hidden_size+hidden_size, hidden_size)

        self.layer2_x = nn.Linear(hidden_size, hidden_size)
        self.layer2_r = nn.Linear(hidden_size, hidden_size)
        self.layer2_tauM = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.layer2_tauAdp = nn.Linear(hidden_size+hidden_size, hidden_size)
   
        self.layer3_x = nn.Linear(hidden_size, output_size)
        self.layer3_tauM = nn.Linear(output_size+output_size, output_size)


        self.act1m = sigmoid_beta(is_train=True)
        self.act1a = sigmoid_beta(is_train=True)
        self.act2m = sigmoid_beta(is_train=True)
        self.act2a = sigmoid_beta(is_train=True)
        self.act3 = sigmoid_beta(is_train=True)
        self.dp = nn.Dropout(0.1)

        nn.init.xavier_normal_(self.layer1_x.weight)
        nn.init.orthogonal_(self.layer1_r.weight)
        # nn.init.xavier_normal_(self.layer1_tauM.weight)
        nn.init.xavier_normal_(self.layer1_tauM.weight)
        nn.init.xavier_normal_(self.layer1_tauAdp.weight)

        nn.init.xavier_normal_(self.layer2_x.weight)
        nn.init.orthogonal_(self.layer2_r.weight)
        # nn.init.xavier_normal_(self.layer1_tauM.weight)
        nn.init.xavier_normal_(self.layer2_tauM.weight)
        nn.init.xavier_normal_(self.layer2_tauAdp.weight)

        nn.init.xavier_normal_(self.layer3_x.weight)
        nn.init.xavier_normal_(self.layer3_tauM.weight)


        nn.init.zeros_(self.layer1_x.bias)
        nn.init.zeros_(self.layer1_tauM.bias)
        nn.init.zeros_(self.layer1_tauAdp.bias)
        nn.init.zeros_(self.layer2_x.bias)
        nn.init.zeros_(self.layer2_tauM.bias)
        nn.init.zeros_(self.layer2_tauAdp.bias)
        nn.init.zeros_(self.layer3_x.bias)
        nn.init.zeros_(self.layer3_tauM.bias)

        # self.bn1a = nn.BatchNorm1d(hidden_size)
        # self.bn1b = nn.BatchNorm1d(hidden_size)
        # self.bn2 = nn.BatchNorm1d(output_size)

        self.bn1a = SeparatedBatchNorm1d(hidden_size,max_length=784)
        self.bn1b = SeparatedBatchNorm1d(hidden_size,max_length=784)
        self.bn2a = SeparatedBatchNorm1d(hidden_size,max_length=784)
        self.bn2b = SeparatedBatchNorm1d(hidden_size,max_length=784)
        self.bn2 = SeparatedBatchNorm1d(output_size,max_length=784)
        self.bn1a.reset_parameters()
        self.bn1b.reset_parameters()
        self.bn2.reset_parameters()
        self.bn1a.bias.data.fill_(0)
        self.bn1b.bias.data.fill_(0)
        self.bn2a.bias.data.fill_(0)
        self.bn2b.bias.data.fill_(0)
        self.bn1a.weight.data.fill_(0.1)
        self.bn1b.weight.data.fill_(0.1)
        self.bn2a.weight.data.fill_(0.1)
        self.bn2b.weight.data.fill_(0.1)
        self.bn2.weight.data.fill_(0.1)
 
        
    def forward(self, inputs, h,i=None):
        self.fr = 0
        # T = inputs.size()[0]
        
        outputs = []
        hiddens = []
 
        b,din,l = inputs.shape
        T = l

        for x_i in range(l):
            if l>700: i = x_i
            x = inputs[:,0,x_i].view(b,1)
    
            dense_x = self.bn1a(self.layer1_x(x),i)+self.bn1b(self.layer1_r(h[1]),i)
           

            tauM1 = self.act1m(self.layer1_tauM(torch.cat((dense_x, h[0]),dim=-1)))
            tauAdp1 = self.act1a(self.layer1_tauAdp(torch.cat((dense_x, h[2]),dim=-1)))
        
            mem_1,spk_1,_,b_1 = mem_update_adp(dense_x, mem=h[0],spike=h[1],
                                            tau_adp=tauAdp1,tau_m=tauM1,b = h[2])

            dense_x2 = self.bn2a(self.layer2_x(spk_1),i)+self.bn2b(self.layer2_r(h[1]),i)
           

            tauM2 = self.act2m(self.layer2_tauM(torch.cat((dense_x2, h[3]),dim=-1)))
            tauAdp2 = self.act2a(self.layer2_tauAdp(torch.cat((dense_x2, h[5]),dim=-1)))
        
            mem_2,spk_2,_,b_2 = mem_update_adp(dense_x2, mem=h[3],spike=h[4],
                                            tau_adp=tauAdp2,tau_m=tauM2,b = h[5])


            
            dense3_x = self.layer3_x(spk_2)
            dense3_x = self.bn2(dense3_x,i)
            tauM2 = self.act3(self.layer3_tauM(torch.cat((dense3_x, h[6]),dim=-1)))
            mem_3 = output_Neuron(dense3_x,mem=h[6],tau_m = tauM2)

            out =mem_3

            h = (mem_1,spk_1,b_1,
                mem_2,spk_2,b_2, 
                mem_3,
                out)

            f_output = F.log_softmax(out, dim=1)
            outputs.append(f_output)
            hiddens.append(h)

            self.fr = self.fr+ (spk_1.detach().cpu().numpy().mean()+spk_2.detach().cpu().numpy().mean())/2.
                
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
        
        # self.l2_loss = nn.MSELoss()

    def forward(self, inputs, hidden):
        # inputs = inputs.permute(2, 0, 1)  
        # print(inputs.shape) # L,B,d

        outputs, hidden, hiddens= self.network.forward(inputs, hidden)

        recon_loss = torch.zeros(1, device=inputs.device)
        return outputs, hidden, recon_loss

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new(bsz,self.nhid).uniform_(),
                weight.new(bsz,self.nhid).zero_(),
                weight.new(bsz,self.nhid).fill_(b_j0),

                weight.new(bsz,self.nhid).uniform_(),
                weight.new(bsz,self.nhid).zero_(),
                weight.new(bsz,self.nhid).fill_(b_j0),

                # layer 3
                weight.new(bsz,self.nout).zero_(),
                # sum spike
                weight.new(bsz,self.nout).zero_(),
                )



