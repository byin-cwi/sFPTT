import os
import shutil
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable


class AddTaskModel(nn.Module):
    def __init__(self, rec_net, hidden_size):
        super(AddTaskModel, self).__init__()
        self.nhid = hidden_size
        self.rnn = rec_net
        self.lin = nn.Linear(hidden_size, 1) 
        self.loss_func = nn.MSELoss()

        nn.init.xavier_normal_(self.lin.weight)

    def forward(self, x, y, h):        
        loss = 0
        output, hidden = self.rnn.forward(x, h)
        out = self.lin(hidden[0])           
        out = out.squeeze()
        y  = y.squeeze()
        loss += self.loss_func(out, y.t())
        return loss, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(1, bsz, self.nhid),
                weight.new_zeros(1, bsz, self.nhid))