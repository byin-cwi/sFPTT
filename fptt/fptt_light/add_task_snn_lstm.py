import os
import shutil
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from snn_models_lstm_type2 import *

"""
python main_addTask_snn.py --alpha 0.5 --beta 0.5 --clip .15 --lr 0.001 --optim 'Adam' --nhid 128 --log-interval 200 --epochs 10000 --parts 10 --bptt 200
"""
class AddTaskModel(nn.Module):
    def __init__(self, input_size,hidden_size):
        super(AddTaskModel, self).__init__()
        self.nhid = hidden_size
        
        # self.lstm_snn = LSTM_cell(input_size,hidden_size)
        self.lstm_snn = nn.LSTMCell(input_size,hidden_size)

        self.lin = nn.Linear(hidden_size, 1) 
        self.loss_func = nn.MSELoss()

        nn.init.xavier_normal_(self.lin.weight)

        self.act1 = nn.Sigmoid()
        self.act2 = nn.Sigmoid()

    def forward(self, x, y, h):        
        loss = 0
        # print(x.shape,y.shape)
        s,b,d = x.shape
        spk_sum = 0
        out = 0
        for i in range(s):    
            # mem_1,c_1,spk_1 = self.lstm_snn(x[i,:,:],h[0],h[1])
            mem_1,c_1 = self.lstm_snn(x[i,:,:],(h[0],h[1]))
            
            h = (mem_1,c_1,out)
            # output, hidden = self.rnn.forward(x, h)
            # spk_sum  = spk_sum + spk_1
        out = self.lin(mem_1)           
        out = out.squeeze()
        y  = y.squeeze()
        # print(y.shape)
        loss += self.loss_func(out, y.t())
        h = (mem_1,c_1,out)
            
        return loss, h

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(bsz, self.nhid),
                weight.new_zeros(bsz, self.nhid),
                weight.new_zeros(bsz, self.nhid),)