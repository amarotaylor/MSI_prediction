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
import numpy as np
import pandas as pd
import pickle
import torch.nn.functional as F
from collections import Counter

# https://github.com/pytorch/accimage
set_image_backend('accimage')
get_image_backend()

# set root dir for images
root_dir = '/n/mounted-data-drive/COAD/'

# normalize and tensorify jpegs
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(),normalize])

sa_train, sa_val = data_utils.process_MSI_data()

train_set = data_utils.TCGADataset_tiles(sa_train, root_dir, transform=transform)
val_set = data_utils.TCGADataset_tiles(sa_val, root_dir, transform=transform)

# set weights for random sampling of tiles such that batches are class balanced
weights = 1.0/np.array(list(Counter(train_set.all_labels).values()),dtype=float)*1e3
reciprocal_weights =[]
for index in range(len(train_set)):
    reciprocal_weights.append(weights[train_set.all_labels[index]])
    
batch_size = 256
sampler = torch.utils.data.sampler.WeightedRandomSampler(reciprocal_weights, len(reciprocal_weights), replacement=True)
train_loader = DataLoader(train_set, batch_size=batch_size, pin_memory=True, sampler=sampler, num_workers=12)
#len(train_set) / batch_size

valid_loader = DataLoader(val_set, batch_size=batch_size, pin_memory=True, num_workers=12)
#len(val_set) / batch_size

resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Linear(2048,2,bias=True)#8192
resnet.cuda()

def embedding_training_loop(e, train_loader, net, criterion, optimizer):
    net.train()
    total_loss = 0
    encoding = torch.tensor([[0,0],[1,0],[1,1]], device='cuda').float()
    
    for idx,(batch,labels) in enumerate(train_loader):
        batch, labels = batch.cuda(), encoding[labels.cuda()]
        output = net(batch)
        loss = criterion(output, labels)
        loss.backward()
        total_loss += loss.detach().cpu().numpy()
        optimizer.step()
        optimizer.zero_grad()
        if idx % 200 == 0:
            print('Epoch: {0}, Batch: {1}, Train NLL: {2:0.4f}'.format(e, idx, loss))
            
    print('Epoch: {0}, Avg Train NLL: {1:0.4f}'.format(e, total_loss/float(idx+1)))
    del batch,labels

def embedding_validation_loop(e, valid_loader, net, criterion, dataset='Val', scheduler=None):
    net.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    encoding = torch.tensor([[0,0],[1,0],[1,1]], device='cuda').float()
    with torch.no_grad():
        for idx,(batch,labels) in enumerate(valid_loader):
            batch, labels = batch.cuda(), encoding[labels.cuda()]
            output = net(batch)
            loss = criterion(output, labels)

            total_loss += loss.detach().cpu().numpy()
            all_labels.extend(torch.sum(labels, dim=1).float().cpu().numpy())
            all_preds.append(torch.argmax(output,1).float().detach().cpu().numpy())

            if idx % 200 == 0:
                print('Epoch: {0}, Batch: {1}, {3} NLL: {2:0.4f}'.format(e, idx, loss, dataset))

        if scheduler is not None:
            scheduler.step(total_loss)
            
    acc = np.mean(np.array([l==p for l,p in zip(all_labels,all_preds)]),dtype=float)
    print('Epoch: {0}, Avg {3} NLL: {1:0.4f}, {3} Acc: {2:0.4f}'.format(e, total_loss/float(idx+1), acc, dataset))
    del batch,labels
    
    return total_loss

learning_rate = 1e-2
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(resnet.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, min_lr=1e-6)

for e in range(1):
    if e % 10 == 0:
        print('---------- LR: {0:0.5f} ----------'.format(optimizer.state_dict()['param_groups'][0]['lr']))
    embedding_training_loop(e, train_loader, resnet, criterion, optimizer)
    val_loss = embedding_validation_loop(e, valid_loader, resnet, criterion, dataset='Val', scheduler=scheduler)