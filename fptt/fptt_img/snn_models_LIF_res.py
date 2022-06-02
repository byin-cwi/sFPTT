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

def save_checkpoint(state, is_best, prefix, filename='_snnres_checkpoint.pth.tar'):
    print('saving at ', prefix+filename)
    torch.save(state, prefix+filename)
    if is_best:
        shutil.copyfile(prefix+filename, prefix+ '_snnlres_model_best.pth.tar')


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


class F_res(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold
        ctx.save_for_backward(input)
        return input.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # temp = abs(input) < lens
        return grad_input * input**2/2. * gamma
        # return grad_input


f_res = F_res.apply

def mem_update_adp(inputs, mem, spike, tau_adp,tau_m, b, dt=1, isAdapt=1):
    alpha = tau_m
    
    ro = tau_adp

    if isAdapt:
        beta = 1.8
    else:
        beta = 0.
    # print('ro: ',ro.shape, '; b:', b.shape, '; spk: ',spike.shape)
    b = ro * b + (1 - ro) * spike
    B = b_j0 + beta * b

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
        self.layer1_tauAdp = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.act1 = nn.Sigmoid()

    def forward(self, x_t, mem_t,spk_t,b_t):    
        if self.is_rec:
            dense_x = self.layer1_x(torch.cat((x_t,spk_t),dim=-1))
        else:
            dense_x = self.layer1_x(x_t)

        tauM1 = self.act1(self.layer1_tauM(torch.cat((dense_x, mem_t),dim=-1)))
        tauAdp1 = self.act1(self.layer1_tauAdp(torch.cat((dense_x,b_t),dim=-1)))

        mem_1,spk_1,_,b_1 = mem_update_adp(dense_x, mem=mem_t,spike=spk_t,
                                        tau_adp=tauAdp1,tau_m=tauM1,b =b_t)

        return mem_1,spk_1,b_1

    def compute_output_size(self):
        return [self.hidden_size]


class SNN_Conv_cell(nn.Module):
    def __init__(self, input_size, output_dim, kernel_size=5,strides=1,padding=0,
                 pooling_type = None, pool_size = 2, pool_strides =2,bias=True):
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

        self.conv1_x = nn.Conv2d(self.input_dim,output_dim,kernel_size=kernel_size,stride=strides,padding=padding,bias=bias)
        self.conv_tauM = nn.Conv2d(output_dim,output_dim,kernel_size=3,stride=1,padding=1)
        self.conv_tauAdp = nn.Conv2d(output_dim,output_dim,kernel_size=3,stride=1,padding=1)
        self.sig1 = nn.Sigmoid()

        self.BN = nn.BatchNorm2d(output_dim)
        self.BN1 = nn.BatchNorm2d(output_dim)
        self.BN2 = nn.BatchNorm2d(output_dim)

        self.output_size = self.compute_output_size()

        nn.init.kaiming_normal_(self.conv1_x.weight)
        nn.init.kaiming_normal_(self.conv_tauM.weight)
        nn.init.kaiming_normal_(self.conv_tauAdp.weight)
        # nn.init.xavier_normal_(self.conv1_x.weight)
        # nn.init.xavier_normal_(self.conv_tauM.weight)
        # nn.init.xavier_normal_(self.conv_tauAdp.weight)
        if bias:
            nn.init.constant_(self.conv1_x.bias,0)
            nn.init.constant_(self.conv_tauM.bias,0)
            nn.init.constant_(self.conv_tauAdp.bias,0)



    def forward(self, x_t, mem_t,spk_t,b_t):
 
        conv_x = self.BN(self.conv1_x(x_t.float()))
        if self.pooling is not None: 
            conv_x = self.pooling(conv_x)

        tauM1 = self.sig1(self.BN1(self.conv_tauM(conv_x+mem_t)))
        tauAdp1 = self.sig1(self.BN2(self.conv_tauAdp(conv_x+b_t)))
       

        mem_1,spk_1,_,b_1 = mem_update_adp(conv_x, mem=mem_t,spike=spk_t,
                                        tau_adp=tauAdp1,tau_m=tauM1,b =b_t)

        return mem_1,spk_1,b_1

    def compute_output_size(self):
        x_emp = torch.randn([1,self.input_size[0],self.input_size[1],self.input_size[2]])   
        out = self.conv1_x(x_emp)
        if self.pooling is not None: 
            out=self.pooling(out)
        return out.shape[1:]

class BasicBlock(nn.Module):
    def __init__(self,in_size,planes,stride=1):
        super(BasicBlock,self).__init__()
        self.network = []
        self.network_size = []
        self.planes = planes
        biased = False

        self.conv1 = SNN_Conv_cell(in_size,planes,3,stride,1,bias=biased)
        in_size1 = self.conv1.compute_output_size()

        self.conv2 = SNN_Conv_cell(in_size1,planes,3,1,1,bias=biased)
        in_size2 = self.conv2.compute_output_size()

        self.network_size = [in_size1,in_size2]
        self.network = [self.conv1, self.conv2]

        self.is_res_conv = 0

        if stride != 1 or in_size[0] != planes:
            self.shortcut = SNN_Conv_cell(in_size,planes,1,stride,0,bias=biased)
            self.network.append(self.shortcut)
            self.network_size.append(in_size2)
            self.is_res_conv = 1
        

    def forward(self,x,h,layer_idx, fr):
        layer_i = layer_idx +1

        h[3*layer_i],h[1+3*layer_i],h[2+3*layer_i]  = self.conv1(x,h[3*layer_i],h[1+3*layer_i],h[2+3*layer_i])
        x_in = h[1+3*layer_i]
        fr += x_in.detach().cpu().numpy().mean()

        layer_i = layer_i+1
        h[3*layer_i],h[1+3*layer_i],h[2+3*layer_i]  = self.conv2(x_in,h[3*layer_i],h[1+3*layer_i],h[2+3*layer_i])
        fr += h[1+3*layer_i].detach().cpu().numpy().mean()

        if self.is_res_conv:
            layer_i = layer_i+1
            h[3*layer_i],h[1+3*layer_i],h[2+3*layer_i]  = self.shortcut(x,h[3*layer_i],h[1+3*layer_i],h[2+3*layer_i])
            fr += h[1+3*layer_i].detach().cpu().numpy().mean()
            x_short_cut = h[1+3*layer_i]
        else:
            x_short_cut = x

        # output = h[1+3*layer_i]*x_short_cut
        output = f_res(h[1+3*layer_i]+x_short_cut)

        return output, layer_i,h, fr
        



class SNN(nn.Module):
    def __init__(self, input_size, hidden_size,output_size, n_timesteps, P=10):
        super(SNN, self).__init__()
        
        print('SNN-ltc CNN ', P)

        # self.res = ['16s1','32s2','64s2']
        # self.res = ['32s1','64s2','128s2']
        # rep c-k-s

        self.res = ['64s1','64s1','128s2','128s1','256s2','256s1','512s2','512s1']
        self.number_blocks = 1
        self.P = P
        self.step = n_timesteps // self.P
        
        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.output_size = output_size
        self.n_timesteps = n_timesteps
        
        self.rnn_name = 'SNN-res'
        
        in_size = input_size
        self.network = []
        self.network_size = []
        self.Blocks = []

        self.conv1 = SNN_Conv_cell(in_size,64,3,1,1) # k-7-p-3
        self.network.append(self.conv1)
        in_size = self.conv1.compute_output_size()
        self.network_size.append(in_size)
        
        for i in range(len(self.res)):
            layer = self.res[i]
            planes = int(layer.split('s')[0])
            stride = int(layer.split('s')[1])
            for _ in range(self.number_blocks):
                a = BasicBlock(in_size,planes,stride)
                self.Blocks.append(a)
                self.network = self.network + a.network
                self.network_size = self.network_size + a.network_size
                in_size = a.network_size[-1]

        # self.network.append(nn.AdaptiveAvgPool2d(in_size[2]))
        self.network = nn.ModuleList(self.network)

        self.avgpool = nn.AvgPool2d(in_size[1],in_size[1])

        self.dp_f = nn.Dropout2d(0.3)
        f_size = int(in_size[0])#*in_size[1]*in_size[2]/16)
   
        # self.layer3_x = nn.Linear(f_size, output_size)
        self.snn1 = SNN_dense_cell(f_size, output_size, is_rec=True)
        self.network_size.append([output_size])

        self.tau_m_o = nn.Parameter(torch.Tensor(output_size))
        self.act3 = nn.Sigmoid()#sigmoid_beta()
        nn.init.normal_(self.tau_m_o, 10.,.1)

        self.dp_in = nn.Dropout2d(0.1)
        

    def forward(self, inputs, h):
        self.fr = 0
        # T = inputs.size()[0]
        
        # outputs = []
        hiddens = []
        h = list(h)
 
        b,c,w,r = inputs.shape

        x_in = inputs.view(b,c,w,r)
        x_in = self.dp_in(x_in)

        layer_i = 0
        h[3*layer_i],h[1+3*layer_i],h[2+3*layer_i]  = self.conv1(x_in,h[3*layer_i],h[1+3*layer_i],h[2+3*layer_i])
        x_in = h[1+3*layer_i]
        self.fr = self.fr+ x_in.detach().cpu().numpy().mean()
        for block_i in range(len(self.Blocks)):
            Block = self.Blocks[block_i]
            x_in, layer_i,h, self.fr = Block.forward(x_in, h, layer_i, self.fr)
        
        x_in = self.avgpool(x_in)
        spk_conv = x_in#self.dp_f(x_in)
        f_spike = torch.flatten(spk_conv,1)

        
        # dense3_x = self.layer3_x(f_spike)
        # tauM2 = self.act3(self.tau_m_o)#self.act3(self.layer3_tauM(torch.cat((dense3_x, h[3]),dim=-1)))
        # h[-2] = output_Neuron(dense3_x,mem=h[-2],tau_m = tauM2)

        layer_i +=1
        h[3*layer_i],h[1+3*layer_i],h[2+3*layer_i]= self.snn1.forward(f_spike,h[3*layer_i],h[1+3*layer_i],h[2+3*layer_i])
        self.fr = self.fr+ h[1+3*layer_i].detach().cpu().numpy().mean()
        # x_in = h[1+3*layer_i]
        h[-2] = h[3*layer_i]



        h[-1] = h[-2]
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
        self.parts = parts # Should be the number of parts

        self.rnn_name = 'SNN resnet'

        self.network = SNN(input_size=ninp, hidden_size=nhid, output_size=nout,n_timesteps=n_timesteps, P=parts)
        self.layer_size = self.network.network_size
        print(self.layer_size)
        
        self.l2_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, hidden):
        # inputs = inputs.permute(2, 0, 1)  
        # print(inputs.shape) # L,B,d
        outputs = []
        if len(inputs.shape)==5:
            b,l,c,w,h = inputs.shape
            
            for i in range(l):
                f_output, hidden, hiddens= self.network.forward(inputs[:,i,:,:,:], hidden)
                outputs.append(f_output)
            recon_loss = torch.zeros(1, device=inputs.device)
        elif len(inputs.shape) == 4:
            # cifar10
            for i in range(self.parts):
                f_output, hidden, hiddens= self.network.forward(inputs, hidden)
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
                states.append(weight.new(bsz,l[0],l[1],l[2]).fill_(b_j0))
            elif len(l) == 1:
                states.append(weight.new(bsz,l[0]).uniform_())
                states.append(weight.new(bsz,l[0]).zero_())
                states.append(weight.new(bsz,l[0]).fill_(b_j0))

        states.append(weight.new(bsz,self.nout).zero_())
        states.append(weight.new(bsz,self.nout).zero_())
        # print(self.layer_size)
        # print([s.shape for s in states])
        return tuple(states)
        # print(self.layer_size)
        # print([s.shape for s in states])
        return tuple(states)



