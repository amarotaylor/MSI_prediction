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
import model_utils
import pickle
import torch.nn.functional as F
from collections import Counter
import argparse


def main():
    parser = argparse.ArgumentParser(description='Model training engine for molecular phenotype models from TCGA images')
    
    parser.add_argument('--Task',help= 'WGD (only implemented task)', default='WGD',required=False, type=str)
    parser.add_argument('--GPU', help='GPU device to use for model training', required=True, type=int)
    parser.add_argument('--n_workers', help='Number of workers to use for dataloaders', required=False, default=12, type=int)
    parser.add_argument('--lr',help='Inital learning rate',required=False, default=1e-4, type=float)
    parser.add_argument('--patience',help='Patience for lr scheduler', required=False, default=10, type=int)
    parser.add_argument('--model_name',help='Path to place saved model state', required=True, type=str)
    parser.add_argument('--batch_size',help='Batch size for training and validation loops',required=False, default=264, type=int)
    parser.add_argument('--epochs',help='Epochs to run training and validation loops', required=False, default=50, type=int)
    parser.add_argument('--magnification',help='Magnification level of tiles', required=False, default='10.0', type=str)
    args = parser.parse_args()
    
    # https://github.com/pytorch/accimage
    set_image_backend('accimage')
    device = torch.device('cuda', args.GPU)
    
    root_dir = data_utils.root_dir_all

    # normalize and tensorify jpegs
    transform_train = train_utils.transform_train
    transform_val = train_utils.transform_validation
    
    # set up model
    input_size = 2048
    hidden_size = 512
    output_size = 1
    state_dict_file = '/n/tcga_models/archive/resnet18_WGD_v03.pt'
    resnet = models.resnet18(pretrained=False)
    resnet.fc = nn.Linear(2048, output_size, bias=True)
    saved_state = torch.load(state_dict_file, map_location=lambda storage, loc: storage)
    resnet.load_state_dict(saved_state)
    
    for p in resnet.parameters():
        p.requires_grad = False
       
    attend_and_pool = model_utils.Attention(input_size, hidden_size, output_size)
    resnet.fc = attend_and_pool
    for p in resnet.fc.parameters():
        p.requires_grad = True
    resnet.cuda(device=device)
    
    optim = torch.optim.Adam(resnet.fc.parameters(), lr = args.lr)
    train_cancers = ['COAD']
    val_cancers = ['COAD']
    pickle_file = '/n/tcga_wgd_sa_all_1.0.pkl'
    batch_all, sa_trains, sa_vals = data_utils.load_COAD_train_val_sa_pickle(pickle_file=pickle_file,
                                                                                       return_all_cancers=True,
                                                                                       split_in_two=False)
    train_idxs = [batch_all.index(cancer) for cancer in train_cancers]    
    val_idxs = [batch_all.index(cancer) for cancer in val_cancers]
    train_sets = []
    val_sets = []
    for i in range(len(train_cancers)):
        train_set = data_utils.TCGA_random_tiles_sampler(sa_trains[batch_all.index(train_cancers[i])], 
                                             root_dir + train_cancers[i] + '/', 
                                             transform=transform_train, 
                                             magnification=args.magnification,tile_batch_size=args.batch_size)
        train_sets.append(train_set)    

    for j in range(len(val_cancers)):
        val_set = data_utils.TCGA_random_tiles_sampler(sa_vals[batch_all.index(val_cancers[j])], 
                                           root_dir + val_cancers[j] + '/', 
                                           transform=transform_val, 
                                           magnification=args.magnification,tile_batch_size=args.batch_size)
        val_sets.append(val_set)
        
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=1,shuffle=True,num_workers=args.n_workers, 
                                            pin_memory=False)

    val_loader = torch.utils.data.DataLoader(val_set,batch_size=1,shuffle=True,num_workers=args.n_workers, 
                                            pin_memory=False)
    
    best_loss = 1e8
    best_acc = 0.0
    criterion=nn.BCEWithLogitsLoss()
    
    for e in range(args.epochs):
        train_utils.training_loop_random_sampling(e,train_loader,device,criterion,resnet,optim,gradient_step_length=10,reporting_step_length=10)
        val_loss,val_acc = train_utils.validation_loop_for_random_sampler(e,val_loader,device,criterion,resnet)
        if val_loss < best_loss:
            torch.save(resnet.state_dict(), args.model_name)
            best_loss = val_loss
            best_acc = val_acc
            print('WROTE MODEL')
        elif val_acc > best_acc:
            torch.save(resnet.state_dict(), args.model_name)
            best_acc = val_acc
            best_loss = val_loss
            print('WROTE MODEL')
            
            
if __name__ == "__main__":
    main()