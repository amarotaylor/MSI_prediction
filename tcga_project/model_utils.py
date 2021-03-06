import torch
import torch.nn as nn
#import copy


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
    
class FeedForward(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, initial_vals=None, dropout=0.0):
        super(FeedForward, self).__init__()
        self.d = nn.Dropout(dropout)
        self.m = nn.ReLU()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
        if initial_vals != None:
            for idx, param in enumerate(self.parameters()):
                param.data = initial_vals[idx].clone()
        
    def forward(self, inputs):
        hidden = self.m(self.linear1(self.d(inputs)))
        output = self.linear2(self.d(hidden))
        return output
    
    def update_params(self, new_vals):
        for idx, param in enumerate(self.parameters()):
            param.data = new_vals[idx].clone()
            
        #self.linear1.weight = torch.nn.Parameter(new_vals[0])
        #self.linear1.bias = torch.nn.Parameter(new_vals[1])
        #self.linear2.weight = torch.nn.Parameter(new_vals[2])
        #self.linear2.bias = torch.nn.Parameter(new_vals[3])
        

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, gated=False):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gated = gated
        self.V = nn.Linear(input_size, hidden_size)
        if self.gated == True:
            self.U = nn.Linear(input_size, hidden_size)
        self.w = nn.Linear(hidden_size, output_size)
        self.sigm = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.sm = nn.Softmax(dim=0)
        self.linear_layer = nn.Linear(input_size,1)
        
    def forward(self, h, return_attention=False):
        if self.gated == True:
            a = self.sm(self.w(self.tanh(self.V(h)) * self.sigm(self.U(h))))
        else:
            a = self.sm(self.w(self.tanh(self.V(h))))
        z = torch.sum(a*h,dim=0)
        logits = self.linear_layer(z)
        if return_attention:
            return logits,a
        else:
            return logits
        
        
class Generator(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.0):
        super(Generator, self).__init__()
        self.d = nn.Dropout(dropout)
        self.m = nn.ReLU()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, inputs):
        hidden = self.d(self.m(self.linear1(inputs)))
        output = self.linear2(hidden)
        return output

    
class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pool_fn, dropout=0.0):
        super(Encoder, self).__init__()
        self.d = nn.Dropout(dropout)
        self.m = nn.ReLU()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.pool_fn = pool_fn
        
    def forward(self, inputs):
        hidden = self.d(self.m(self.linear1(inputs)))
        hidden = self.pool_fn(hidden)
        output = self.linear2(hidden)
        return output
