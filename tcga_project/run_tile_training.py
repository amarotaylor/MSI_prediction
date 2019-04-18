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
import argparse

def main():
    parser = argparse.ArgumentParser(description='Model training engine for molecular phenotype models from TCGA images')
    
    parser.add_argument('--Task',help='WGD prediction or MSI (only implemented tasks)', required=True,type=str)
    parser.add_argument('--GPU', help='GPU device to use for model training', required=True,type=int)
    parser.add_argument('--n_workers', help='Number of workers to use for dataloaders', required=False, default = 12,type=int)
    parser.add_argument('--lr',help='Inital learning rate',required = False, default=1e-4,type=float)
    parser.add_argument('--patience',help='Patience for lr scheduler', required = False, default =10,type=int)
    parser.add_argument('--model_name',help='Path to place saved model state', required = True,type=str)
    parser.add_argument('--batch_size',help='Batch size for training and validation loops', required=False,default=264,type=int)
    parser.add_argument('--epochs',help='Epochs to run training and validation loops', required=False,default=50,type=int)

    args = parser.parse_args()
    
    # https://github.com/pytorch/accimage
    set_image_backend('accimage')
    device = torch.device('cuda', args.GPU)
    # set root dir for images
    root_dir = data_utils.root_dir_coad

    # normalize and tensorify jpegs
    transform = train_utils.transform
    
    # set the task
    # TODO: implement a general for for table to perform predictions
    if args.Task.upper() == 'MSI':
        sa_train, sa_val = data_utils.process_MSI_data()
        output_shape = 2
    elif args.Task.upper() == 'WGD':
        sa_train, sa_val = data_utils.process_WGD_data()
        output_shape = 1
    
    train_set = data_utils.TCGADataset_tiles(sa_train, root_dir, transform=transform)
    val_set = data_utils.TCGADataset_tiles(sa_val, root_dir, transform=transform)

    # set weights for random sampling of tiles such that batches are class balanced
    weights = 1.0/np.array(list(Counter(train_set.all_labels).values()),dtype=float)*1e3
    reciprocal_weights =[]
    for index in range(len(train_set)):
        reciprocal_weights.append(weights[train_set.all_labels[index]])

    batch_size = args.batch_size
    # current WeightedRandomSampler is too slow when replacement = False. 
    # TODO: implement switch to weighted loss or weighted sampler
    sampler = torch.utils.data.sampler.WeightedRandomSampler(reciprocal_weights, len(reciprocal_weights), replacement=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, pin_memory=True, sampler=sampler, num_workers=args.n_workers)
    valid_loader = DataLoader(val_set, batch_size=batch_size, pin_memory=True, num_workers=args.n_workers)
    
    # TODO: allow resnet model specification or introduce other model choices
    resnet = models.resnet18(pretrained=True)
    # TODOD: implement flexible solution to these hardcoded values
    
    resnet.fc = nn.Linear(2048,output_shape,bias=True)#8192
    resnet.cuda(device=device)

    learning_rate = args.lr
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(resnet.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, min_lr=1e-6)

    best_loss = 1e8
    for e in range(args.epochs):
        if e % 10 == 0:
            print('---------- LR: {0:0.5f} ----------'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        train_utils.embedding_training_loop(e, train_loader, resnet, criterion, optimizer,device=device)
        val_loss = train_utils.embedding_validation_loop(e, valid_loader, resnet, criterion, dataset='Val', scheduler=scheduler,device=device)
        if val_loss < best_loss:
            torch.save(resnet.state_dict(),args.model_name)
            best_loss = val_loss

if __name__ == "__main__":
    main()
