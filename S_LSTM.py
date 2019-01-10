#coding=utf-8
"""
@version=1.0
@author:zsh
@file:model.py
@time:2019/1/8 19:47
"""
import torch as t
from torch import nn
import torch.nn.functional as F
MAX_LENGTH = 10
use_cuda=t.cuda.is_available()
class Encoder(nn.Module):
    def __init__(self,input_size,hidden_size,n_layers=1):
        super(Encoder,self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size,hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size)
    def forward(self, input,hidden):
        input = input.unsqueeze(1)
        embedded = self.embedding(input)  # batch, hidden
        output = embedded.permute(1, 0, 2)
        for i in range(self.n_layers):
            output,hidden = self.gru(output,hidden)
        return output,hidden
    def initHidden(self):
        result = t.zeros(1,1,self.hidden_size)
        return result.cuda() if use_cuda else result

class Decoder(nn.Module):
    def __init__(self,hidden_size,output_size,n_layers=1):
        super(Decoder,self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size,hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size)
        self.out = nn.Linear(hidden_size,output_size)
        self.softmax = nn.LogSoftmax()
    def forward(self, input,hidden):
        # input=input.unsqueeze(1)
        output =  self.embedding(input)
        output = output.permute(1,0,2) # permute the dimession of tensor
        for i in range(self.n_layers):
            # output,hidden = self.gru(output,hidden)
            output=F.relu(output)
            output,hidden = self.gru(output,hidden)
        output = self.softmax(self.out(output[0]))
        return output,hidden
    def initHidden(self):
        result = t.zeros(1,1,self.hidden_size)
        return result.cuda() if use_cuda else result


