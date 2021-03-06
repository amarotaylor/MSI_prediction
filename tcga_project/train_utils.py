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
    
    
def calc_tile_acc_stats(labels, preds, all_types=None, all_jpgs=None):
    # total acc
    acc = np.mean(np.array(labels) == np.array(preds))
    # acc by label
    d = {'label': labels, 'pred': preds}
    df = pd.DataFrame(data = d)
    df['correct_tile'] = df['label'] == df['pred']
    tile_acc_by_label = ', '.join([str(int(i)) + ': ' + str(float(df.groupby(['label'])['correct_tile'].mean()[int(i)]))[:6] \
                                   for i in np.unique(df['label'])])    
    if all_types is not None: # cancer_type and jpg_to_sample are separate
        d = {'label': labels, 'pred': preds, 'type': all_types, 'sample': all_jpgs}
        df = pd.DataFrame(data = d)
        df2 = df.groupby(['type','sample'])['label','pred'].mean().round()
        df2['correct_sample'] = df2['label'] == df2['pred']
        mean_pool_acc = df2['correct_sample'].mean()
        slide_acc_by_label=', '.join([str(int(i))+': '+str(float(df2.groupby(['label'])['correct_sample'].mean()[int(i)]))[:6] \
                                        for i in np.unique(df2['label'])])
        df3 = df.groupby(['type','sample'])['label','pred'].max().round()
        df3['correct_sample'] = df3['label'] == df3['pred']
        max_pool_acc = df3['correct_sample'].mean()
        slide_acc_by_mlabel=', '.join([str(int(i))+': '+str(float(df3.groupby(['label'])['correct_sample'].mean()[int(i)]))[:6] \
                                       for i in np.unique(df3['label'])])
        return acc, tile_acc_by_label, mean_pool_acc, slide_acc_by_label, max_pool_acc, slide_acc_by_mlabel
    elif all_jpgs is not None: # cancer_type and jpg_to_sample are combined
        jpgs = torch.cat(all_jpgs)
        d = {'label': labels, 'pred': preds, 'type': jpgs[:,0], 'sample': jpgs[:,1]}
        df = pd.DataFrame(data = d)
        df2 = df.groupby(['type','sample'])['label','pred'].mean().round()
        df2['correct_sample'] = df2['label'] == df2['pred']
        mean_pool_acc = df2['correct_sample'].mean()
        slide_acc_by_label=', '.join([str(int(i))+': '+str(float(df2.groupby(['label'])['correct_sample'].mean()[int(i)]))[:6] \
                                        for i in np.unique(df2['label'])])
        return acc, tile_acc_by_label, mean_pool_acc, slide_acc_by_label
    else:
        return acc, tile_acc_by_label
    
    
def maml_train_local(step, tiles, labels, resnet, local_models, alpha=0.01, criterion=nn.BCEWithLogitsLoss(),device=torch.device('cuda',0)):
    resnet.eval()
    idx = int(tiles.shape[0] / 2)
    num_tasks = int(tiles.shape[1])
    
    # grads storage    
    grads = [torch.zeros(p.shape).cuda(device=device) for p in local_models[0].parameters()]

    #t = torch.randint(num_tasks, (1,)).item()
    for t in range(num_tasks):
        # first forward pass, step
        net = local_models[t]
        net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr = alpha)

        inputs = tiles[:idx,t,:,:,:]
        embed = resnet(inputs)
        output = net(embed)
        loss = criterion(output, labels[:idx,t].unsqueeze(1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # second forward pass, store grads
        inputs = tiles[idx:,t,:,:,:]
        embed = resnet(inputs)
        output = net(embed)
        loss = criterion(output, labels[idx:,t].unsqueeze(1))
        loss.backward()
        grads[0] = grads[0] + net.linear1.weight.grad.data
        grads[1] = grads[1] + net.linear1.bias.grad.data
        grads[2] = grads[2] + net.linear2.weight.grad.data
        grads[3] = grads[3] + net.linear2.bias.grad.data
        optimizer.zero_grad()
        
    if step % 100 == 0:
        output = (output.contiguous().view(-1) > 0.5).float().detach().cpu().numpy()
        labels = labels[idx:,t].contiguous().view(-1).float().detach().cpu().numpy()
        acc, tile_acc_by_label = calc_tile_acc_stats(labels, output)
        print('Step: {0}, Train NLL: {1:0.4f}, Acc: {2:0.4f}, By Label: {3}'.format(step, loss, acc, tile_acc_by_label))

    return grads, local_models


def maml_train_global(theta_global, model_global, grads, eta=0.01):
    theta_global = [theta_global[i] - (eta * grads[i]) for i in range(len(theta_global))]
    model_global.update_params(theta_global)
    #model_global.linear1.weight = torch.nn.Parameter(theta_global[0])
    #model_global.linear1.bias = torch.nn.Parameter(theta_global[1])
    #model_global.linear2.weight = torch.nn.Parameter(theta_global[2])
    #model_global.linear2.bias = torch.nn.Parameter(theta_global[3])

    return theta_global, model_global


def maml_validate_all(e, resnet, model_global, val_loaders, device=torch.device('cuda',0), criterion=nn.BCEWithLogitsLoss()):
    resnet.eval()
    model_global.eval()
    
    total_loss = 0
    all_output = []
    all_labels = []
    all_types = []
    all_jpgs = []
    
    total_steps = 0
    divisor = 100
    for idx,val_loader in enumerate(val_loaders):
        for step,(batch,labels,jpg_to_sample) in enumerate(val_loader):
            inputs, labels = batch.cuda(device=device), labels.cuda(device=device).view(-1,1).float()

            embed = resnet(inputs)
            output = model_global(embed)
            loss = criterion(output, labels)
            
            prop_pos = torch.mean(labels)
            #weighted_loss = (torch.sum(loss[labels == 1.0]) / prop_pos) + (torch.sum(loss[labels == 0.0]) / (1 - prop_pos))
            total_loss += loss.detach().cpu().numpy()
            
            output = (output.contiguous().view(-1) > 0.5).float().detach().cpu().numpy()
            labels = labels.contiguous().view(-1).detach().cpu().numpy()
            jpg_to_sample = jpg_to_sample.view(-1).float().numpy()
            
            all_output.extend(output)
            all_labels.extend(labels)
            all_types.extend([idx] * batch.shape[0])
            all_jpgs.extend(jpg_to_sample)

            if step % divisor == 0:
                acc, tile_acc_by_label = calc_tile_acc_stats(labels, output)
                print('Step: {0}, Val NLL: {1:0.4f}, Acc: {2:0.4f}, By Label: {3}'.format(total_steps, loss, acc,
                                                                                          tile_acc_by_label))
                total_steps += divisor
                
    acc, tile_acc_by_label, mean_pool_acc, slide_acc_by_label, max_pool_acc, slide_acc_by_mlabel = \
    calc_tile_acc_stats(all_labels, all_output, all_types=all_types, all_jpgs=all_jpgs)
    print('Epoch: {0}, Val NLL: {1:0.4f}, Tile-Level Acc: {2:0.4f}, By Label: {3}'.format(e, total_loss / (float(len(val_loaders[0])) + float(len(val_loaders[1])) + float(len(val_loaders[2]))), acc, tile_acc_by_label))
    print('------ Slide-Level Acc (Mean-Pooling): {0:0.4f}, By Label: {1}'.format(mean_pool_acc, slide_acc_by_label))
    print('------ Slide-Level Acc (Max-Pooling): {0:0.4f}, By Label: {1}'.format(max_pool_acc, slide_acc_by_mlabel))
    return total_loss, acc, mean_pool_acc



def validation_loop_for_random_sampler(e,val_loader,device,criterion,resnet,scheduler=None):
    pred_batch = []
    true_label = []
    torch.cuda.empty_cache()
    loss = torch.tensor(0.0,device=device)
    with torch.no_grad():
        for step,(slide, label, coords) in enumerate(val_loader):
            slide,label = slide.squeeze(0).cuda(device),label.cuda(device)
            logits,_ = resnet(slide)
            loss += criterion(logits,label.float())
            pred_batch.append(torch.sigmoid(logits).detach().cpu().numpy()>0.5)
            true_label.append(label.detach().cpu().numpy())
            
            del slide,label,logits,_
            torch.cuda.empty_cache()
    #scheduler.step(loss)
    pred_batch = np.array(pred_batch)
    true_label = np.array(true_label)
    acc = np.mean(pred_batch==true_label)
    acc_1 = np.mean(pred_batch[true_label==1])
    acc_0 = np.mean(1-pred_batch[true_label==0])
    if scheduler != None:
        scheduler.step(loss)
    loss = loss.detach().cpu().numpy()
    print('Epoch: {0}, Val Mean NLL: {1:0.4f}, Val Accuracy: {2:0.2f} \
           Class Accuracy: WGD = {3:0.2f}, Diploid = {4:0.2f}'\
              .format(e,loss/step,acc,acc_1,acc_0))
    return loss,acc
    
    
def training_loop_random_sampling(e,train_loader,device,criterion,resnet,optimizer,gradient_step_length=3,reporting_step_length=10):
    grads = []
    track_loss = torch.tensor(0.0,device=device)
    for p in resnet.fc.parameters():
        grads.append(torch.zeros_like(p.data,device=device))
        
    for step,(slide, label, coords) in enumerate(train_loader):
        optimizer.zero_grad()
        slide,label = slide.squeeze(0).cuda(device),label.cuda(device)
        logits,_ = resnet(slide)
        loss = criterion(logits,label.float())
        loss.backward()
      
        for ix,p in enumerate(resnet.fc.parameters()):
            grads[ix] += p.grad.detach().clone()
            
        track_loss += loss.detach().clone()    
        optimizer.zero_grad()
        if step%gradient_step_length ==0 and step>0:
            for ix,p in enumerate(resnet.fc.parameters()):
                p.grad.data = grads[ix]/gradient_step_length
                grads[ix] = torch.zeros_like(p.data,device=device)
            optimizer.step()
            optimizer.zero_grad()
                
        if step%reporting_step_length == 0 and step>0:
            print('Epoch: {0}, Step: {1}, Train NLL: {2:0.4f}'.format(e, step, track_loss.detach().cpu().numpy()/reporting_step_length))
            track_loss = 0.0
    del slide, label, loss, logits, _
    torch.cuda.empty_cache()
    
    
    
    
def training_loop_pooled_embeddings(e,step_size,optimizer,net,train_embeddings,train_jpgs_to_slide
                                    ,train_labels,criterion,n_samples,sample_weight):
    logits_vec = torch.zeros((step_size+1,1)).cuda()
    labels_vec = torch.zeros_like(logits_vec).cuda()
    batch_idx = 0
    track_loss = 0    
    idxs_train = np.linspace(0,n_samples,n_samples+1,dtype=int)
    idexs = np.random.choice(idxs_train,size=n_samples.numpy(),p=sample_weight)
    
    for idx in idexs:
        slide = train_embeddings[train_jpgs_to_slide==idx]
        labels_vec[batch_idx] = train_labels[train_jpgs_to_slide==idx].unique().float().cuda()
        logits_vec[batch_idx] = torch.mean(net(slide))
        if batch_idx == step_size:
            loss = criterion(logits_vec,labels_vec)
            loss.backward()
            track_loss += loss.detach().cpu()            
            optimizer.step()
            optimizer.zero_grad()
            logits_vec = torch.zeros((step_size+1,1)).cuda()
            labels_vec = torch.zeros_like(logits_vec).cuda()
            batch_idx = 0
        else:
            batch_idx += 1
        
    loss = criterion(logits_vec,labels_vec)
    loss.backward()
    track_loss += loss.detach().cpu() * (batch_idx/step_size)
    track_loss = track_loss * (float(step_size) / float(n_samples))
    optimizer.step()
    optimizer.zero_grad()
    print('Epoch: {0}, Train NLL: {1:0.4f}'.format(e,track_loss.numpy()))
    
    
def validation_loop_pooled_embeddings(e,scheduler,net,val_embeddings,val_jpgs_to_slide
                                    ,val_labels,criterion,n_samples):
    logits_vec = torch.zeros((n_samples+1,1)).cuda()
    labels_vec = torch.zeros_like(logits_vec).cuda()
    track_loss = 0
    idexs = np.linspace(0,n_samples,n_samples+1,dtype=int)
    with torch.no_grad():
        for idx in idexs:
            slide = val_embeddings[val_jpgs_to_slide==idx]
            labels_vec[idx] = val_labels[val_jpgs_to_slide==idx].unique().float().cuda()
            logits_vec[idx] = torch.mean(net(slide))
        loss = criterion(logits_vec,labels_vec)
        scheduler.step(loss)
        mask = labels_vec.cpu().numpy() == (logits_vec>0.5).float().cpu().numpy()
        acc = np.mean(mask)
        acc_1 = np.mean(mask[labels_vec.cpu().numpy()==1.])
        acc_0 = np.mean(mask[labels_vec.cpu().numpy()==0.])
        print('Epoch: {0}, Validation NLL: {1:0.4f}, Total Acc: {2:0.3f}, Acc by label; diploid:{3:0.3f} WGD:{4:0.3f}'.format(e,loss.cpu().numpy(),acc,acc_0,acc_1))
        return loss,acc
            
