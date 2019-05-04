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
    
    parser.add_argument('--Task',help='WGD-ALL, WGD, MSI, MSI-SINGLE_LABEL (only implemented tasks)', required=True, type=str)
    parser.add_argument('--GPU', help='GPU device to use for model training', required=True, type=int)
    parser.add_argument('--n_workers', help='Number of workers to use for dataloaders', required=False, default=12, type=int)
    parser.add_argument('--lr',help='Inital learning rate',required=False, default=1e-4, type=float)
    parser.add_argument('--patience',help='Patience for lr scheduler', required=False, default=10, type=int)
    parser.add_argument('--model_name',help='Path to place saved model state', required=True, type=str)
    parser.add_argument('--batch_size',help='Batch size for training and validation loops',required=False, default=264, type=int)
    parser.add_argument('--epochs',help='Epochs to run training and validation loops', required=False, default=50, type=int)
    parser.add_argument('--magnification',help='Magnification level of tiles', required=False, default='5.0', type=str)
    parser.add_argument('--fine_tune_classifier_only',help='Freeze convolutional layers',action='store_true')
    args = parser.parse_args()
    
    # https://github.com/pytorch/accimage
    set_image_backend('accimage')
    device = torch.device('cuda', args.GPU)
    
    # set root dir for images
    if args.Task.upper() == 'WGD-ALL':
        root_dir = data_utils.root_dir_all
    else:
        root_dir = data_utils.root_dir_coad

    # normalize and tensorify jpegs
    transform_train = train_utils.transform_train
    transform_val = train_utils.transform_validation
    
    # set the task
    # TODO: implement a general for table to perform predictions
    if args.Task.upper() == 'WGD-ALL':
        pickle_file = '/n/tcga_wgd_sa_all_1.0.pkl'
        batch_all, sa_trains, sa_vals = data_utils.load_COAD_train_val_sa_pickle(pickle_file=pickle_file,
                                                                                       return_all_cancers=True,
                                                                                       split_in_two=False)
        train_cancers = ['COAD', 'BRCA', 'READ_10x', 'LUSC_10x', 'BLCA_10x', 'LUAD_10x', 'STAD_10x', 'HNSC_10x']
        train_idxs = [batch_all.index(cancer) for cancer in train_cancers]
        
        val_cancers = ['COAD', 'BRCA', 'READ_10x', 'LUSC_10x', 'BLCA_10x', 'LUAD_10x', 'STAD_10x', 'HNSC_10x']
        val_idxs = [batch_all.index(cancer) for cancer in val_cancers]
        
        sa_train = {}
        sa_val = {}
        ct_train = []
        ct_val = []
        
        for idx, (sa_t, sa_v) in enumerate(zip(sa_trains, sa_vals)):
            if idx in train_idxs:
                sa_train.update(sa_t)                
                ct_train.extend([batch_all[idx]] * len(list(sa_trains[idx].keys())))                
            if idx in val_idxs:
                sa_val.update(sa_v)
                ct_val.extend([batch_all[idx]] * len(list(sa_vals[idx].keys())))
                
        train_set = data_utils.TCGADataset_tiles(sa_train, root_dir, transform=transform_train, 
                                                 magnification=args.magnification, all_cancers=True, cancer_type=ct_train)
        val_set = data_utils.TCGADataset_tiles(sa_val, root_dir, transform=transform_val, 
                                               magnification=args.magnification, all_cancers=True, cancer_type=ct_val)
        
        jpg_to_sample = val_set.jpg_to_sample  
        output_shape = 1
    else:
        if args.Task.upper() == 'MSI':
            sa_train, sa_val = data_utils.process_MSI_data()
            output_shape = 2
        elif args.Task.upper() == 'WGD':
            sa_train, sa_val = data_utils.process_WGD_data()
            output_shape = 1
        elif args.Task.upper() == 'MSI-SINGLE_LABEL':
            sa_train, sa_val = data_utils.process_MSI_data()
            # replace ordinal labels with binary
            for key,value in zip(sa_train.keys(),sa_train.values()):
                sa_train[key] = int(value>=1)            
            for key,value in zip(sa_val.keys(),sa_val.values()):
                sa_val[key] = int(value>=1)
            output_shape = 1

        # save sample_annotations_train, sample_annotations_val as pickle
        pickle_file = args.model_name[:-3] + '_sa.pkl'
        with open(pickle_file, 'wb') as f: 
            pickle.dump([sa_train, sa_val], f)

        train_set = data_utils.TCGADataset_tiles(sa_train, root_dir, transform=transform_train, magnification=args.magnification)
        val_set = data_utils.TCGADataset_tiles(sa_val, root_dir, transform=transform_val, magnification=args.magnification)
        jpg_to_sample = val_set.jpg_to_sample
    
    # set weights for random sampling of tiles such that batches are class balanced
    counts = [c[1] for c in sorted(Counter(train_set.all_labels).items())]
    weights = 1.0 / np.array(counts, dtype=float) * 1e3
    reciprocal_weights =[]
    for index in range(len(train_set)):
        reciprocal_weights.append(weights[train_set.all_labels[index]])

    batch_size = args.batch_size
    # current WeightedRandomSampler is too slow when replacement = False. 
    # TODO: implement switch to weighted loss or weighted sampler
    #sampler = torch.utils.data.sampler.WeightedRandomSampler(reciprocal_weights, len(reciprocal_weights), replacement=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=args.n_workers)
    valid_loader = DataLoader(val_set, batch_size=batch_size, pin_memory=True, num_workers=args.n_workers)
    
    learning_rate = args.lr
    # TODO: allow resnet model specification or introduce other model choices
    resnet = models.resnet18(pretrained=True)
    # TODO: implement flexible solution to these hardcoded values
    if args.fine_tune_classifier_only:
        for param in resnet.parameters():
            param.requires_grad = False
    resnet.fc = nn.Linear(2048,output_shape,bias=True)#8192
    resnet.cuda(device=device)

    if args.fine_tune_classifier_only:
        optimizer = torch.optim.Adam(resnet.fc.parameters(), lr = learning_rate)
    else:
        optimizer = torch.optim.Adam(resnet.parameters(), lr = learning_rate)
        
    criterion_train = nn.BCEWithLogitsLoss(reduction = 'mean')
    criterion_val = nn.BCEWithLogitsLoss(reduction = 'none')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, min_lr=1e-6)

    best_loss = 1e8
    best_acc = 0.0
    for e in range(args.epochs):
        if e % 10 == 0:
            print('---------- LR: {0:0.8f} ----------'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        train_utils.embedding_training_loop(e, train_loader, resnet, criterion_train, 
                                            optimizer,device=device,task=args.Task.upper())
        val_loss, val_acc = train_utils.embedding_validation_loop(e, valid_loader, resnet, criterion_val, 
                                                                  jpg_to_sample, dataset='Val', scheduler=scheduler,
                                                                  device=device, task=args.Task.upper())
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
