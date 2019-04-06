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