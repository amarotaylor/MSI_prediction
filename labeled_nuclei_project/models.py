import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        

class Generator(nn.Module):
    def __init__(self, n_conv_layers, kernel_size, n_conv_filters, hidden_size, n_rnn_layers, dropout=0.5):
        super(Generator, self).__init__()
        self.n_conv_layers = n_conv_layers
        self.kernel_size = kernel_size
        self.n_conv_filters = n_conv_filters
        self.hidden_size = hidden_size
        self.n_rnn_layers = n_rnn_layers
        self.conv_layers = []
        self.m = nn.MaxPool2d(2, stride=2)
        self.relu = nn.ReLU()
         
        in_channels = 3        
        for layer in range(self.n_conv_layers):
            self.conv_layers.append(nn.Conv2d(in_channels, self.n_conv_filters[layer], self.kernel_size[layer]))
            self.conv_layers.append(self.relu)
            self.conv_layers.append(self.m)
            in_channels = self.n_conv_filters[layer]
        self.conv = nn.Sequential(*self.conv_layers)
        in_channels = in_channels * 25

        self.lstm = nn.LSTM(in_channels, self.hidden_size, self.n_rnn_layers, batch_first=True, 
                            dropout=dropout, bidirectional=True) 
        in_channels = hidden_size * 2
        self.classification_layer = nn.Linear(in_channels, 2)
        
    def forward(self, x):
        embed = self.conv(x)
        embed = embed.view(1,x.shape[0],-1)
        self.lstm.flatten_parameters()
        output, hidden = self.lstm(embed)
        y = self.classification_layer(output)
        return y
    
    def zero_grad(self):
        """Sets gradients of all model parameters to zero."""
        for p in self.parameters():
            if p.grad is not None:
                p.grad.data.zero_()
                
                
                
                
def update_tile_shape(H_in, W_in, kernel_size, dilation = 1., padding = 0., stride = 1.):
    H_out = (H_in + 2. * padding - dilation * (kernel_size-1) -1)/stride + 1
    W_out = (W_in + 2. * padding - dilation * (kernel_size-1) -1)/stride + 1
    return int(np.floor(H_out)),int(np.floor(W_out))

class Neighborhood_Generator(nn.Module):
    def __init__(self, n_conv_layers, n_fc_layers, kernel_size, n_conv_filters, hidden_size, dropout=0.5,
                 dilation = 1., padding = 0, H_in = 27, W_in = 27):
        super(Neighborhood_Generator, self).__init__()
        # set class attributes
        self.n_conv_layers = n_conv_layers
        self.kernel_size = kernel_size
        self.n_conv_filters = n_conv_filters
        self.hidden_size = hidden_size
        self.n_fc_layers = n_fc_layers
        self.conv_layers = []
        self.fc_layers = []
        self.n = nn.Dropout(dropout)
        self.m = nn.MaxPool2d(2, stride=2)
        self.relu = nn.ReLU()
        self.H_in,self.W_in  = H_in,W_in
        
        # perform the encoding 
        in_channels = 3        
        for layer in range(self.n_conv_layers):
            self.conv_layers.append(nn.Conv2d(in_channels, self.n_conv_filters[layer], self.kernel_size[layer]))
            self.conv_layers.append(self.relu)
            self.conv_layers.append(self.m)
            # convolution
            self.H_in,self.W_in = update_tile_shape(self.H_in, self.W_in, kernel_size[layer])
            # max pooling
            self.H_in,self.W_in = update_tile_shape(self.H_in, self.W_in, 2, stride = 2)
            in_channels = self.n_conv_filters[layer]
        
        # compute concatenation size
        in_channels = in_channels * self.H_in * self.W_in * 5
        
        # infer the z
        for layer in range(self.n_fc_layers):
            self.fc_layers.append(nn.Linear(in_channels, self.hidden_size[layer]))
            self.fc_layers.append(self.relu)
            self.fc_layers.append(self.n)
            in_channels = self.hidden_size[layer]
            
        self.conv = nn.Sequential(*self.conv_layers)    
        self.fc = nn.Sequential(*self.fc_layers)
        self.classification_layer = nn.Linear(in_channels, 2)
        
    def forward(self, x, neighbors):
        embed = self.conv(x)
        
        embed = embed.view(x.shape[0],-1)
        e_neighbors = [torch.index_select(embed,0,n) for n in neighbors]
        embed_n = torch.stack([torch.cat([e.unsqueeze(0),n],0).view(-1) for e,n in zip(embed,e_neighbors)])
        output = self.fc(embed_n)
        logits = self.classification_layer(output)
        return logits