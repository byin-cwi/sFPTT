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

# from tutorial.utils.util import *
from nets.yolo_layers import *

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

def save_checkpoint(state, is_best, prefix, filename='_snn_yolov4_checkpoint.pth.tar'):
    print('saving at ', prefix+filename)
    torch.save(state, prefix+filename)
    if is_best:
        shutil.copyfile(prefix+filename, prefix+ '_snnl_yolov4_model_best.pth.tar')


def count_parameters(model):
    return sum(p.numel() for p in model.network.parameters() if p.requires_grad)



class SNN(nn.Module):
    def __init__(self, input_size,P=10):
        super(SNN, self).__init__()
        
        print('SNN-ltc yolo ', P)
        self.P = P
        self.step =1
        # self.anchors = [[10,13],[16,30],[33,23]]
        self.anchors =[ [[81,82], [135,169], [344,319]] , [[10,14], [23,27], [37,58]] ]
        self.num_classes = 20

        self.input_size = input_size
        
        self.rnn_name = 'SNN-yolo_v4'

        in_size = input_size
        self.network = []
        self.network_size = []


        # 416,416,3 -> 208,208,32 
        self.conv1 = SNN_Conv_cell(in_size, 32, kernel_size=3, strides=2)
        in_size = self.conv1.compute_output_size()
        self.network.append(self.conv1)
        self.network_size.append(in_size)

        # 208,208,32 -> 104,104,64
        self.conv2 = SNN_Conv_cell(in_size, 64, kernel_size=3, strides=2)
        in_size = self.conv2.compute_output_size()
        self.network.append(self.conv2)
        self.network_size.append(in_size)

        # 104,104,64 -> 52,52,128
        self.block1 = BasicBlock(in_size,64)
        self.network = self.network + self.block1.network
        self.network_size = self.network_size + self.block1.network_size
        in_size = [128,52,52] 

        # 52,52,128 -> 26,26,256
        self.block2 = BasicBlock(in_size,128)
        self.network = self.network + self.block2.network
        self.network_size = self.network_size + self.block2.network_size
        in_size = [256,26,26]

        # 26,26,256 -> 13,13,512 # output feature 1
        self.block3 = BasicBlock(in_size,256)
        self.network = self.network + self.block3.network
        self.network_size = self.network_size + self.block3.network_size
        in_size = [512,13,13]

        # 13,13,256 -> 13,13,512 # output feature 2
        self.conv3 = SNN_Conv_cell(in_size, 512, kernel_size=3)
        in_size = self.conv3.compute_output_size()
        self.network.append(self.conv3)
        self.network_size.append(in_size)

        # 13,13, 512 -> 13,13,256
        self.conv_P5 = SNN_Conv_cell(in_size, 256, kernel_size=1, strides=1)
        in_size = self.conv_P5.compute_output_size()
        self.network.append(self.conv_P5)
        self.network_size.append(in_size)

        # 13,13, 256 -> 13,13,512
        self.conv_yolo_1 = SNN_Conv_cell(in_size, 512, kernel_size=3, strides=1)
        in_size = self.conv_yolo_1.compute_output_size()
        self.network.append(self.conv_yolo_1)
        self.network_size.append(in_size)

        # 13,13, 512 -> 13,13,255 # yolo0
        self.conv_P5_yolo_1 = nn.Conv2d(512, 75, kernel_size=1, stride=1)
        self.yolo_tau1 = nn.Parameter(torch.Tensor([75]))
        nn.init.normal_(self.yolo_tau1, 4.,1)
        self.act = nn.Sigmoid()
    

        # 13,13,256 -> 26,26,128
        in_size =[256, 13, 13] 
        self.conv_up = SNN_Conv_cell(in_size, 128, kernel_size=1,pooling_type='up')
        in_size = self.conv_up.compute_output_size()
        self.network.append(self.conv_up)
        self.network_size.append(in_size)

        #cat [26,26,256] + [26,26,128]{feature1} -> [26,26,384]
        # [26,26,384] -> [26,26,256]
        in_size = [384, 26,26]
        self.conv_yolo_2 = SNN_Conv_cell(in_size, 256, kernel_size=3,strides=1)
        in_size = self.conv_yolo_2.compute_output_size()
        self.network.append(self.conv_yolo_2)
        self.network_size.append(in_size)

        # [26,26,256] -> [26,26,255]
        self.conv_P5_yolo_2 = nn.Conv2d(256, 75, kernel_size=1, stride=1)
        self.yolo_tau2 = nn.Parameter(torch.Tensor([75]))
        nn.init.normal_(self.yolo_tau2, 4.,1)


        self.network = nn.ModuleList(self.network)
     
        
        self.dp_in = nn.Dropout2d(0.1)

    def forward(self, inputs, h):
        self.fr = 0
        h = list(h)
 
        b,c,w,r = inputs.shape
        x_in = inputs.view(b,c,w,r)
        x_in = self.dp_in(x_in)

        layer_i = 0
        h[2*layer_i],h[1+2*layer_i],_  = self.conv1(x_in,h[2*layer_i],h[1+2*layer_i])
        x_in = h[1+2*layer_i]
        self.fr = self.fr+ x_in.detach().cpu().numpy().mean()

        layer_i = 1
        h[2*layer_i],h[1+2*layer_i],_  = self.conv2(x_in,h[2*layer_i],h[1+2*layer_i])
        x_in = h[1+2*layer_i]
        self.fr = self.fr+ x_in.detach().cpu().numpy().mean()

        # layer_i = 2
        x_in,_,layer_i,h,self.fr  = self.block1(x_in,h,layer_i,self.fr)

        x_in,_,layer_i,h,self.fr  = self.block2(x_in,h,layer_i,self.fr)

        x_in,feat_1,layer_i,h,self.fr  = self.block3(x_in,h,layer_i,self.fr) # feat1

        layer_i += 1
        h[2*layer_i],h[1+2*layer_i],_  = self.conv3(x_in,h[2*layer_i],h[1+2*layer_i])
        x_in = h[1+2*layer_i]
        self.fr = self.fr+ x_in.detach().cpu().numpy().mean()

        layer_i += 1
        h[2*layer_i],h[1+2*layer_i],_  = self.conv_P5(x_in,h[2*layer_i],h[1+2*layer_i])
        x_in = h[1+2*layer_i]
        P5 = h[1+2*layer_i]
        self.fr = self.fr+ x_in.detach().cpu().numpy().mean()

        layer_i += 1
        h[2*layer_i],h[1+2*layer_i],_  = self.conv_yolo_1(x_in,h[2*layer_i],h[1+2*layer_i])
        x_in = h[1+2*layer_i]
        self.fr = self.fr+ x_in.detach().cpu().numpy().mean()


        
        h[-2] = self.conv_P5_yolo_1(x_in) # YOLO 1
        # h[-2] = output_Neuron(self.conv_P5_yolo_1(x_in),h[-2],self.act(self.yolo_tau1))


        layer_i += 1
        # print(P5.shape,h[2*layer_i].shape,h[-2].shape,x_in.shape,h[1+2*(layer_i-3)].shape)
        h[2*layer_i],h[1+2*layer_i],_  = self.conv_up(P5,h[2*layer_i],h[1+2*layer_i])
        x_in = h[1+2*layer_i]
        self.fr = self.fr+ x_in.detach().cpu().numpy().mean()

        # cat ,
        # print(x_in.shape,feat_1.shape)
        x_in  = torch.cat((x_in,feat_1),dim=1)
        

        layer_i += 1
        h[2*layer_i],h[1+2*layer_i],_  = self.conv_yolo_2(x_in,h[2*layer_i],h[1+2*layer_i])
        x_in = h[1+2*layer_i]
        self.fr = self.fr+ x_in.detach().cpu().numpy().mean()


        h[-1] = self.conv_P5_yolo_2(x_in) # YOLO 2
        # h[-1] = output_Neuron(self.conv_P5_yolo_2(h[1+2*layer_i]),h[-1],self.act(self.yolo_tau2))



        h = tuple(h)
        self.fr = self.fr/(len(self.network)+1.)
                
        final_state = h

        return [h[-2],h[-1]], final_state,self.fr

class YoloBody(nn.Module):
    def __init__(self, arch, nout, parts=10, pretrained=False):

        super(YoloBody, self).__init__()


        self.rnn_name = 'SNN yolo'
        self.p = parts

        self.network = SNN(input_size=[3,416,416], P=parts)
        self.layer_size = self.network.network_size
        print(self.layer_size)

    def forward(self, inputs,hidden):
        # outputs = []
        # hidden = self.init_hidden(inputs.shape[0])
        # for i in range(self.p):
        output, hidden,_= self.network.forward(inputs, hidden)
            # outputs.append(output)
        # output = outputs[-2]
  
        return output[0], output[1]

    def init_hidden(self, bsz):

        weight = next(self.parameters()).data
        states = []
        for l in self.layer_size:
            if len(l) == 3:
                states.append(weight.new(bsz,l[0],l[1],l[2]).uniform_())
                states.append(weight.new(bsz,l[0],l[1],l[2]).zero_())
                # states.append(weight.new(bsz,l[0],l[1],l[2]).fill_(b_j0))
            elif len(l) == 1:
                states.append(weight.new(bsz,l[0]).uniform_())
                states.append(weight.new(bsz,l[0]).zero_())
                # states.append(weight.new(bsz,l[0]).fill_(b_j0))
        
        states.append(weight.new(bsz,75,13,13).zero_())
        states.append(weight.new(bsz,75,26,26).zero_())
        # states.append(weight.new(bsz,5,22,22,25).zero_())
        # states.append(weight.new(bsz,self.nout).zero_())
        return tuple(states)
     



