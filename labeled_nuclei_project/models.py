import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, n_conv_layers, n_fc_layers, kernel_size, n_conv_filters, hidden_size, dropout=0.5):
        super(ConvNet, self).__init__()
        self.n_conv_layers = n_conv_layers
        self.n_fc_layers = n_fc_layers
        self.kernel_size = kernel_size
        self.n_conv_filters = n_conv_filters
        self.hidden_size = hidden_size
        self.conv_layers = []
        self.fc_layers = []
        self.m = nn.MaxPool2d(2, stride=2)
        self.n = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        in_channels = 3        
        for layer in range(self.n_conv_layers):
            self.conv_layers.append(nn.Conv2d(in_channels, self.n_conv_filters[layer], self.kernel_size[layer]))
            self.conv_layers.append(self.relu)
            self.conv_layers.append(self.m)
            in_channels = self.n_conv_filters[layer]
        in_channels = in_channels * 25
        for layer in range(self.n_fc_layers):
            self.fc_layers.append(nn.Linear(in_channels, self.hidden_size[layer]))
            self.fc_layers.append(self.relu)
            self.fc_layers.append(self.n)
            in_channels = self.hidden_size[layer]
        self.conv = nn.Sequential(*self.conv_layers)
        self.fc = nn.Sequential(*self.fc_layers)
        self.classification_layer = nn.Linear(in_channels, 2)
        
    def forward(self, x):
        embed = self.conv(x)
        embed = embed.view(x.shape[0],-1)
        y = self.fc(embed)
        return y
    
    
class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, gated=True):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gated = gated
        self.V = nn.Linear(input_size, hidden_size)
        self.U = nn.Linear(input_size, hidden_size)
        self.w = nn.Linear(hidden_size, output_size)
        self.sigm = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.sm = nn.Softmax(dim=0)
        
    def forward(self, h):
        if self.gated == True:
            a = self.sm(self.w(self.tanh(self.V(h)) * self.sigm(self.U(h))))
        else:
            a = self.sm(self.w(self.tanh(self.V(h))))
        return a
    
    
class pool(nn.Module):
    def __init__(self,attn = None):
        super(pool,self).__init__()
        self.attn = attn
    def forward(self,x):
        if self.attn == None:
            return torch.mean(x,0)
        else:
            a = self.attn(x)
            v = torch.transpose(a, dim0=0, dim1=1).matmul(x)
            return v.squeeze(0)