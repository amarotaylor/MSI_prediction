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
transform_train = transforms.Compose([transforms.ToTensor(),
                                      transforms.ToPILImage(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ColorJitter(hue=0.02,saturation=0.1),
                                      transforms.ToTensor(),normalize])


def embedding_training_loop(e, train_loader, net, criterion, optimizer,device='cuda:0', task = 'MSI'):
    net.train()
    total_loss = 0
    if task == 'MSI':
        encoding = torch.tensor([[0,0],[1,0],[1,1]], device=device).float()
    elif task == 'WGD-ALL' or task == 'WGD' or task == 'MSI-SINGLE_LABEL':
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
    elif task == 'WGD-ALL' or task == 'WGD' or task == 'MSI-SINGLE_LABEL':
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
    tile_acc_by_label = ', '.join([str(i) + ': ' + str(float(df.groupby(['label'])['correct_tile'].mean()[i]))[:6] for i in range(encoding.shape[0])])
    
    df2 = df.groupby(['sample'])['label','pred'].mean().round()
    df2['correct_sample'] = df2['label'] == df2['pred']
    mean_pool_acc = df2['correct_sample'].mean()
    
    df3 = df.groupby(['sample'])['label','pred'].max()
    df3['correct_sample'] = df3['label'] == df3['pred']
    max_pool_acc = df3['correct_sample'].mean()
    
    slide_acc_by_label = ', '.join([str(i) + ': ' + str(float(df2.groupby(['label'])['correct_sample'].mean()[i]))[:6] for i in range(encoding.shape[0])])
    
    print('Epoch: {0}, Avg {3} NLL: {1:0.4f}, Median {3} NLL: {2:0.4f}'.format(e, total_loss/(float(idx+1) * batch.shape[0]), 
                                                                               np.median(all_loss), dataset))
    print('------ {2} Tile-Level Acc: {0:0.4f}; By Label: {1}'.format(acc, tile_acc_by_label, dataset))
    print('------ {2} Slide-Level Acc: Mean-Pooling: {0:0.4f}, Max-Pooling: {1:0.4f}'.format(mean_pool_acc, max_pool_acc, 
                                                                                             dataset))
    print('------ {1} Slide-Level Acc (Mean-Pooling) By Label: {0}'.format(slide_acc_by_label, dataset))
    
    del batch,labels
    return total_loss, mean_pool_acc


def tcga_embedding_training_loop(e, train_loader, resnet, slide_level_classification_layer, criterion, 
                                 optimizer, pool_fn, slide_batch_size=10, tile_batch_size=256):
    slide_level_classification_layer.train()
    
    total_loss = 0
    logits_list = []
    labels_list = []
    for idx, (image, label, coords) in enumerate(train_loader):
        if idx % slide_batch_size != 0 or idx == 0:
            image, label = image.squeeze(0), label.float().cuda()
            #tiles = torch.utils.data.TensorDataset(image)
            #tile_loader = DataLoader(tiles, batch_size=tile_batch_size, shuffle=False, pin_memory=True, num_workers=5)
            #n_batches = image.shape[0] // tile_batch_size + 1   
            #tile_loader = [image[b*tile_batch_size:(b+1)*tile_batch_size,:,:,:] for b in range(n_batches)]
            all_emb = []
            #for tile in tile_loader:
            if len(image.shape) < 5:
                image = image.unsqueeze(0)
            for tile in image:
                #tile = tile[0].cuda()
                tile = tile.cuda()
                emb = resnet(tile)
                all_emb.append(emb)
            embed = torch.cat(all_emb)
            output = pool_fn(embed)

            logits = slide_level_classification_layer(output)
            logits_list.append(logits)
            labels_list.append(label)
        else:
            loss = criterion(torch.stack(logits_list), torch.stack(labels_list))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print('Epoch: {0}, Step: {1}, Train Batch NLL: {2:0.4f}'.format(e, idx, loss.detach().cpu().numpy()))
            logits_list = []
            labels_list = []
            

def tcga_embedding_validation_loop(e, valid_loader, resnet, slide_level_classification_layer, criterion, pool_fn, 
                                   tile_batch_size=256, scheduler=None, dataset='Val'):
    slide_level_classification_layer.eval()
    
    total_loss = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for idx, (image, label, coords) in enumerate(valid_loader):
            image, label = image.squeeze(0), label.float().cuda()
            tiles = torch.utils.data.TensorDataset(image)
            tile_loader = DataLoader(tiles, batch_size=tile_batch_size, shuffle=False, pin_memory=True)

            all_emb = []
            for tile in tile_loader:
                tile = tile[0].cuda()
                emb = resnet(tile)
                all_emb.append(emb)
            embed = torch.cat(all_emb)
            output = pool_fn(embed)

            logits = slide_level_classification_layer(output)
            loss = criterion(logits, label)
            total_loss += loss.detach().cpu().numpy()
            all_labels.append(label.float().cpu().numpy())
            all_preds.append((torch.sigmoid(logits) > 0.5).float().detach().cpu().numpy())

            if idx % 10 == 0:
                print('Epoch: {0}, Step: {1}, {3} Slide NLL: {2:0.4f}'.format(e, idx, loss.detach().cpu().numpy(), 
                                                                              dataset))
                
    scheduler.step(total_loss)
    acc = np.mean(np.array(all_labels) == np.array(all_preds))
    print('Epoch: {0}, {3} Avg NLL: {1:0.4f}, {3} Accuracy: {2:0.4f}'.format(e, total_loss/float(idx+1), 
                                                                             acc, dataset))
    return total_loss




def tcga_tiled_slides_training_loop(e, train_loader, resnet, 
                                    slide_level_classification_layer, criterion, 
                                 optimizer, pool_fn,device='cuda:0',train_set=None,p_step =0.0001):
    # track number of slides seen
    p_update = torch.tensor(0.,device=device)
    slide_level_classification_layer.train()
    # store embeddings, labels, and memberships
    embeddings = []
    slide_membership = []
    step = 0
    batches = 0
    slide_labels = torch.tensor(train_set.sample_labels, device=device)
    for batch,labels,coords,idxs in train_loader:
        # get embeddings
        batch,labels,coords,idxs = batch.cuda(),labels.cuda(),coords.cuda(),idxs.cuda()
        if len(embeddings) == 0:
            current_slide = torch.min(idxs)
        # append each batched results
        embeddings.extend(resnet(batch))
        slide_membership.extend(idxs)

        p_update += p_step
        batches+=1

        if torch.rand(1,device=device) < p_update:
            slide_membership = torch.stack(slide_membership)
            slides = torch.unique(slide_membership,sorted=True)
            embeddings = torch.stack(embeddings)
            labels = torch.index_select(slide_labels,0,slides)
            pooled = torch.stack([pool_fn(
                    torch.masked_select(
                        embeddings,(slide_membership == slide).view(-1,1)
                    ).view(-1,2048)) for slide in slides])
            logits = slide_level_classification_layer(pooled)
            loss = criterion(logits,labels.float().view(-1,1))    
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()    
            embeddings = []
            print('Epoch: {0}, Step: {1}, Slide Number: {2}, Train Batch NLL: {3:0.4f}'.format(e, step, torch.max(slide_membership).detach().cpu().numpy(), loss.detach().cpu().numpy()))

            slide_membership = []
            step+=1
            batches = 0
            p_update = torch.tensor(0.,device=device)
    slide_membership = torch.stack(slide_membership)
    slides = torch.unique(slide_membership,sorted=True)
    embeddings = torch.stack(embeddings)
    labels = torch.index_select(slide_labels,0,slides)
    pooled = torch.stack([pool_fn(
                    torch.masked_select(
                        embeddings,(slide_membership == slide).view(-1,1)
                    ).view(-1,2048)) for slide in slides])
    logits = slide_level_classification_layer(pooled)
    loss = criterion(logits,labels.float().view(-1,1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad() 
    print('Epoch: {0}, Step: {1}, Slide Number: {2}, Train Batch NLL: {3:0.4f}'.format(e, step, torch.max(slide_membership).detach().cpu().numpy(), loss.detach().cpu().numpy()))
    del slide_membership, embeddings, labels, pooled, slides, batch, coords
    
    
def tcga_tiled_slides_validation_loop(e, val_loader, resnet, 
                                    slide_level_classification_layer, criterion, 
                                 scheduler, pool_fn,device='cuda:0',val_set=None):
    slide_level_classification_layer.eval()
    with torch.no_grad():
        # store embeddings, labels, and memberships
        embeddings = []
        slide_membership = []
        step = 0
        batches = 0
        slide_labels = torch.tensor(val_set.sample_labels, device=device)
        for batch,labels,coords,idxs in val_loader:
            # get embeddings
            batch,labels,coords,idxs = batch.cuda(),labels.cuda(),coords.cuda(),idxs.cuda()
            if len(embeddings) == 0:
                current_slide = torch.min(idxs)
            # append each batched results
            embeddings.extend(resnet(batch))
            slide_membership.extend(idxs)



        slide_membership = torch.stack(slide_membership)
        slides = torch.unique(slide_membership,sorted=True)
        embeddings = torch.stack(embeddings)
        labels = torch.index_select(slide_labels,0,slides)
        pooled = torch.stack([pool_fn(
                        torch.masked_select(
                            embeddings,(slide_membership == slide).view(-1,1)
                        ).view(-1,2048)) for slide in slides])
        logits = slide_level_classification_layer(pooled)
        loss = criterion(logits,labels.float().view(-1,1))
        all_preds = (torch.sigmoid(logits) > 0.5).float().detach().cpu().numpy()
        acc = np.mean(labels.detach().cpu().numpy() == all_preds)
        scheduler.step(loss)
        print('Epoch: {0}, Step: {1}, Slide Number: {2}, Val Avg NLL: {3:0.4f}, Val Accuracy: {4:0.2f}'.format(e, step, torch.max(slide_membership).detach().cpu().numpy(), loss.detach().cpu().numpy(),acc))
        del slide_membership, embeddings, labels, pooled, slides, batch, coords

        return loss, acc