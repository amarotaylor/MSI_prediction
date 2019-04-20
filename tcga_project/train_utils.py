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


# normalize and tensorify jpegs
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_validation = transforms.Compose([transforms.ToTensor(),
                                normalize])
transform_train = transforms.Compose([transforms.ToTensor(),transforms.ToPILImage(),
                                transforms.RandomVerticalFlip(),transforms.RandomHorizontalFlip(),
                               transforms.ColorJitter(hue=0.02,saturation=0.1),
                                      transforms.ToTensor(),normalize])


def embedding_training_loop(e, train_loader, net, criterion, optimizer,device='cuda:0', task = 'MSI'):
    net.train()
    total_loss = 0
    if task == 'MSI':
        encoding = torch.tensor([[0,0],[1,0],[1,1]], device=device).float()
    elif task == 'WGD':
        encoding = torch.tensor([[0],[1]], device=device).float()
    
    for idx,(batch,labels) in enumerate(train_loader):
        batch, labels = batch.cuda(device=device), encoding[labels.cuda(device=device)]
        output = net(batch)
        loss = criterion(output, labels)
        loss.backward()
        total_loss += loss.detach().cpu().numpy()
        optimizer.step()
        optimizer.zero_grad()
        if idx % 100 == 0 and idx > 0:
            print('Epoch: {0}, Batch: {1}, Train NLL: {2:0.4f}'.format(e, idx, loss))
  
    print('Epoch: {0}, Avg Train NLL: {1:0.4f}'.format(e, total_loss/float(idx+1)))
    del batch,labels

def embedding_validation_loop(e, valid_loader, net, criterion, jpg_to_sample, 
                              dataset='Val', scheduler=None, device='cuda:0', task='MSI'):
    net.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    all_loss = []
    
    if task == 'MSI':
        encoding = torch.tensor([[0,0],[1,0],[1,1]], device=device).float()
    elif task == 'WGD':
        encoding = torch.tensor([[0],[1]], device=device).float()
        
    with torch.no_grad():
        for idx,(batch,labels) in enumerate(valid_loader):
            batch, labels = batch.cuda(device=device), encoding[labels.cuda(device=device)]
            output = net(batch)
            loss = criterion(output, labels)
        
            total_loss += torch.sum(loss.detach().mean(dim=1)).cpu().numpy()
            all_labels.extend(torch.sum(labels, dim=1).float().cpu().numpy())
            all_preds.extend(torch.sum(torch.sigmoid(output) > 0.5, dim=1).float().detach().cpu().numpy())
            all_loss.extend(loss.detach().mean(dim=1).cpu().numpy())
            
            if idx % 100 == 0 and idx > 0:
                print('Epoch: {0}, Batch: {1}, {3} NLL: {2:0.4f}'.format(e, idx, 
                                                                         torch.sum(loss.detach())/batch.shape[0], dataset))

        if scheduler is not None:
            scheduler.step(total_loss)
            
    acc = np.mean(np.array(all_labels) == np.array(all_preds))
    
    d = {'label': all_labels, 'pred': all_preds, 'sample': jpg_to_sample}
    df = pd.DataFrame(data = d)
    df['correct_tile'] = df['label'] == df['pred']
    df.groupby(['label'])['correct_tile'].mean()
    acc_label = ', '.join([str(i) + ': ' + str(df.groupby(['label'])['correct_tile'].mean()[i])[:6] for i in range(encoding.shape[0])])
    
    df2 = df.groupby(['sample'])['label','pred'].mean().round()
    df2['correct_sample'] = df2['label'] == df2['pred']
    mean_pool_acc = df2['correct_sample'].mean()
    
    df3 = df.groupby(['sample'])['label','pred'].max()
    df3['correct_sample'] = df3['label'] == df3['pred']
    max_pool_acc = df3['correct_sample'].mean()
    
    print('Epoch: {0}, Avg {3} NLL: {1:0.4f}, Median {3} NLL: {2:0.4f}'.format(e, total_loss/(float(idx+1) * batch.shape[0]), 
                                                                               np.median(all_loss), dataset))
    print('------ {2} Tile-Level Acc: {0:0.4f}; By Label: {1}'.format(acc, acc_label, dataset))
    print('------ {2} Slide-Level Acc: Mean-Pooling: {0:0.4f}, Max-Pooling: {1:0.4f}'.format(mean_pool_acc, max_pool_acc, 
                                                                                             dataset))
    
    del batch,labels
    return total_loss, mean_pool_acc