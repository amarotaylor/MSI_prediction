import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def embedding_training_loop(e, train_loader, net, criterion, optimizer,pool_fn):
    net.train()
    total_loss = 0
    
    for slide,label in train_loader:
        slide.squeeze_()
        slide, label = slide.cuda(), label.cuda()
        output = net(slide)
        pool = pool_fn(output).unsqueeze(0)
        output = net.classification_layer(pool)
        loss = criterion(output, label)
        loss.backward()
        total_loss += loss.detach().cpu().numpy()
        optimizer.step()
        optimizer.zero_grad()
        
    print('Epoch: {0}, Train NLL: {1:0.4f}'.format(e, total_loss))
    

def embedding_validation_loop(e, valid_loader, net, criterion,pool_fn):
    net.eval()
    total_loss = 0
    labels = []
    preds = []
    
    for slide,label in valid_loader:
        slide.squeeze_()
        slide, label = slide.cuda(), label.cuda()
        output = net(slide)
        pool = pool_fn(output).unsqueeze(0)
        output = net.classification_layer(pool)
        loss = criterion(output, label)
        
        total_loss += loss.detach().cpu().numpy()
        labels.extend(label.float().cpu().numpy())
        preds.append(torch.argmax(output).float().detach().cpu().numpy())
    
    acc = np.mean(np.array(labels) == np.array(preds))
    print('Epoch: {0}, Val NLL: {1:0.4f}, Val Acc: {2:0.4f}'.format(e, total_loss, acc))
    
    return total_loss


def instance_training_loop(e, train_loader, net, criterion, optimizer,pool_fn):
    net.train()
    total_loss = 0
    
    for slide,label in train_loader:
        slide.squeeze_()
        slide, label = slide.cuda(), label.cuda()
        output = net(slide)
        output = net.classification_layer(output)
        output = pool_fn(output).unsqueeze(0)
        loss = criterion(output, label)
        loss.backward()
        total_loss += loss.detach().cpu().numpy()
        optimizer.step()
        optimizer.zero_grad()
        
    print('Epoch: {0}, Train NLL: {1:0.4f}'.format(e, total_loss))
    

def instance_validation_loop(e, valid_loader, net, criterion,pool_fn):
    net.eval()
    total_loss = 0
    labels = []
    preds = []
    
    for slide,label in valid_loader:
        slide.squeeze_()
        slide, label = slide.cuda(), label.cuda()
        output = net(slide)
        output = net.classification_layer(output)
        output = pool_fn(output).unsqueeze(0)
        loss = criterion(output, label)
        
        total_loss += loss.detach().cpu().numpy()
        labels.extend(label.float().cpu().numpy())
        preds.append(torch.argmax(output).float().detach().cpu().numpy())
    
    acc = np.mean(np.array(labels) == np.array(preds))
    print('Epoch: {0}, Val NLL: {1:0.4f}, Val Acc: {2:0.4f}'.format(e, total_loss, acc))
    
    return total_loss