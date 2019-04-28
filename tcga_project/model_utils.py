import torch
import torch.nn as nn


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
            self.linear1.weight = torch.nn.Parameter(initial_vals[0])
            self.linear1.bias = torch.nn.Parameter(initial_vals[1])
            self.linear2.weight = torch.nn.Parameter(initial_vals[2])
            self.linear2.bias = torch.nn.Parameter(initial_vals[3])
        
    def forward(self, inputs):
        hidden = self.m(self.linear1(self.d(inputs)))
        output = self.linear2(self.d(hidden))
        return output
    
    def update_params(self, new_vals):
        self.linear1.weight = torch.nn.Parameter(new_vals[0])
        self.linear1.bias = torch.nn.Parameter(new_vals[1])
        self.linear2.weight = torch.nn.Parameter(new_vals[2])
        self.linear2.bias = torch.nn.Parameter(new_vals[3])