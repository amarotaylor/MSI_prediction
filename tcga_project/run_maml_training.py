import os
import gc
import sys
import torch
import psutil
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn import metrics
from collections import Counter
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import models, set_image_backend
import data_utils
import train_utils
import model_utils
import argparse


def main():
    parser = argparse.ArgumentParser(description='MAML training engine for molecular phenotype models from TCGA images')
    
    parser.add_argument('--GPU', help='GPU device to use for model training', required=True, type=int)
    parser.add_argument('--n_workers', help='Number of workers to use for dataloaders', required=False, default=12, type=int)
    parser.add_argument('--alpha',help='Inital learning rate for local training',required=False, default=1e-4, type=float)
    parser.add_argument('--eta',help='Inital learning rate for global training',required=False, default=1e-4, type=float)
    parser.add_argument('--patience',help='Patience for lr scheduler', required=False, default=10, type=int)
    parser.add_argument('--model_name',help='Path to place saved model state', required=True, type=str)
    parser.add_argument('--batch_size',help='Batch size for training and validation loops',required=False, default=264, type=int)
    parser.add_argument('--epochs',help='Epochs to run training and validation loops', required=False, default=50, type=int)
    parser.add_argument('--magnification',help='Magnification level of tiles', required=False, default='5.0', type=str)
    args = parser.parse_args()
    
    # setup
    set_image_backend('accimage')
    device = torch.device('cuda', args.GPU)
        
    # load sample annotations pickle
    pickle_file = '/home/sxchao/MSI_prediction/tcga_project/tcga_wgd_sa_all.pkl'
    batch_all, _, _, sa_trains, sa_vals = data_utils.load_COAD_train_val_sa_pickle(pickle_file=pickle_file,
                                                                                   return_all_cancers=True, 
                                                                                   split_in_two=True)
    # normalize and tensorify jpegs
    train_transform = train_utils.transform_train
    val_transform = train_utils.transform_validation
    
    # initialize Datasets
    train_sets = []
    val_sets = []

    train_cancers = ['COAD', 'BRCA', 'READ_10x', 'LUSC_10x', 'BLCA_10x', 'LUAD_10x', 'STAD_10x', 'HNSC_10x']
    val_cancers = ['UCEC', 'LIHC_10x', 'KIRC_10x']
    
    magnification = args.magnification
    root_dir = '/n/mounted-data-drive/'
    for i in range(len(train_cancers)):
        train_set = data_utils.TCGADataset_tiles(sa_trains[batch_all.index(train_cancers[i])], 
                                                 root_dir + train_cancers[i] + '/', 
                                                 transform=train_transform, 
                                                 magnification=magnification, 
                                                 batch_type='tile')
        train_sets.append(train_set)    

    for j in range(len(val_cancers)):
        val_set = data_utils.TCGADataset_tiles(sa_vals[batch_all.index(val_cancers[j])], 
                                               root_dir + val_cancers[j] + '/', 
                                               transform=val_transform, 
                                               magnification=magnification, 
                                               batch_type='tile')
        val_sets.append(val_set)
    
    # get DataLoaders    
    batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(data_utils.ConcatDataset(*train_sets), 
                                           batch_size=batch_size, 
                                           shuffle=True, 
                                           num_workers=args.n_workers, 
                                           pin_memory=True)

    val_loader = torch.utils.data.DataLoader(data_utils.ConcatDataset(*val_sets, return_jpg_to_sample=True), 
                                            batch_size=batch_size, 
                                            shuffle=True, 
                                            num_workers=args.n_workers, 
                                            pin_memory=True)

    # model args
    state_dict_file = '/n/tcga_models/resnet18_WGD_all_10x.pt'
    input_size = 2048
    hidden_size = 512
    output_size = 1
    
    # initialize trained resnet
    resnet = models.resnet18(pretrained=False)
    resnet.fc = nn.Linear(2048, output_size, bias=True)
    saved_state = torch.load(state_dict_file, map_location=lambda storage, loc: storage)
    resnet.load_state_dict(saved_state)

    # freeze layers
    resnet.fc = model_utils.Identity()
    resnet.cuda(device=device)
    for param in resnet.parameters():
        param.requires_grad = False
    
    # initialize theta_global
    model_global = model_utils.FeedForward(input_size, hidden_size, output_size).cuda()
    theta_global = []
    for p in model_global.parameters():
        theta_global.append(torch.randn(list(p.shape)).cuda())

    model_global.linear1.weight = torch.nn.Parameter(theta_global[0])
    model_global.linear1.bias = torch.nn.Parameter(theta_global[1])
    model_global.linear2.weight = torch.nn.Parameter(theta_global[2])
    model_global.linear2.bias = torch.nn.Parameter(theta_global[3])

    # initialize local models, set theta_local = theta_global    
    local_models = []
    for i in range(len(train_cancers)):
        local_models.append(model_utils.FeedForward(input_size, hidden_size, output_size, theta_global).cuda()) 
    
    # training params
    num_epochs = args.epochs
    alpha = args.alpha
    eta = args.eta
    patience = args.patience
    factor = 0.1
    patience_count = 0
    previous_loss = 1e8
    best_loss = 1e8
    best_acc = 0.0

    # train meta-learner
    for e in range(num_epochs):
        # reduce LR on plateau
        if patience_count > patience:
            alpha = factor * alpha
            eta = factor * eta
            patience_count = 0
            print('--- LR DECAY --- Alpha: {0:0.8f}, Eta: {1:0.8f}'.format(alpha, eta))

        for step, (tiles, labels) in enumerate(train_loader):  
            tiles, labels = tiles.cuda(), labels.cuda().float()           
            grads, local_models = train_utils.maml_train_local(step, tiles, labels, resnet, local_models, alpha = alpha)
            theta_global, model_global = train_utils.maml_train_global(theta_global, model_global, grads, eta = eta)
            for i in range(len(local_models)):
                local_models[i].update_params(theta_global)

        loss, acc, mean_pool_acc = train_utils.maml_validate(e, resnet, model_global, val_loader)

        if loss > previous_loss:
            patience_count += 1
        else:
            patience_count = 0
        previous_loss = loss
        
        if loss < best_loss:
            torch.save(model_global.state_dict(), args.model_name)
            best_loss = val_loss
        elif mean_pool_acc > best_acc:
            torch.save(model_global.state_dict(), args.model_name)
            best_acc = val_acc

if __name__ == "__main__":
    main()