import os
import sys
import torch
import torch.nn as nn
import accimage
from PIL import Image
from imageio import imread
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms, set_image_backend, get_image_backend
import data_utils
import train_utils
import numpy as np
import pandas as pd
import pickle
import torch.nn.functional as F
from collections import Counter

# https://github.com/pytorch/accimage
set_image_backend('accimage')
#get_image_backend()

# set root dir for images
root_dir = data_utils.root_dir_coad

# normalize and tensorify jpegs
transform = train_utils.transform

sa_train, sa_val = data_utils.process_MSI_data()
train_set = data_utils.TCGADataset_tiles(sa_train, root_dir, transform=transform)
val_set = data_utils.TCGADataset_tiles(sa_val, root_dir, transform=transform)

# set weights for random sampling of tiles such that batches are class balanced
weights = 1.0/np.array(list(Counter(train_set.all_labels).values()),dtype=float)*1e3
reciprocal_weights =[]
for index in range(len(train_set)):
    reciprocal_weights.append(weights[train_set.all_labels[index]])
    
batch_size = 356
sampler = torch.utils.data.sampler.WeightedRandomSampler(reciprocal_weights, len(reciprocal_weights), replacement=False)
train_loader = DataLoader(train_set, batch_size=batch_size, pin_memory=True, sampler=sampler, num_workers=16)
valid_loader = DataLoader(val_set, batch_size=batch_size, pin_memory=True, num_workers=16)

resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Linear(2048,2,bias=True)#8192
resnet.cuda()

learning_rate = 1e-4
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(resnet.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, min_lr=1e-6)

best_loss = 1e8
for e in range(200):
    if e % 10 == 0:
        print('---------- LR: {0:0.5f} ----------'.format(optimizer.state_dict()['param_groups'][0]['lr']))
    train_utils.embedding_training_loop(e, train_loader, resnet, criterion, optimizer)
    val_loss = train_utils.embedding_validation_loop(e, valid_loader, resnet, criterion, dataset='Val', scheduler=scheduler)
    if val_loss < best_loss:
        torch.save(resnet.state_dict(),'resnet.pt')
        best_loss = val_loss

