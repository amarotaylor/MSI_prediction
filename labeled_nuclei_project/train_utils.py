import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

lsm = nn.LogSoftmax(dim=2)

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


def sampler(slide, gen, num_samples):
    zis = []
    grads = []
    all_grads = []
    for p in gen.parameters():
        start = [num_samples]
        start.extend(list(p.shape))
        all_grads.append(torch.zeros(start, device='cuda'))
        grads.append(torch.zeros(p.shape, device='cuda'))
        
    for sample in range(num_samples):
        preds = gen(slide)
        logits = lsm(preds).squeeze(0)
        b = torch.distributions.bernoulli.Bernoulli(logits=logits[:,1])
        zi = b.sample() #zis = b.sample(torch.Size([batch_size]))
        zis.append(zi)

        logprobs = b.log_prob(zi).sum()
        logprobs.backward()

        for idx,p in enumerate(gen.parameters()):
            all_grads[idx][sample] = p.grad

        gen.zero_grad()
        
    return zis, grads, all_grads


def rationales_training(e, train_loader, gen, enc, pool_fn, num_samples, lamb1, lamb2, xent,
                        learning_rate, optimizer):
    gen.train()
    enc.train()
    
    total_loss = 0
    for slide,label in train_loader:
        slide,label = slide.squeeze(0).cuda(),label.cuda()
        zis, grads, all_grads = sampler(slide, gen, num_samples)
        zis = torch.stack(zis)
        
        rationales = [slide[zi==1,:,:,:] for zi in zis]
        sampled_rationales = torch.cat(rationales,dim=0)
        outputs = enc(sampled_rationales)
        
        lens = zis.sum(dim=1)
        indexs = torch.cat([torch.zeros(1),torch.cumsum(lens,0).cpu()]).int()
        outputs = [outputs[indexs[n]:indexs[n+1]] for n,ix in enumerate(indexs[:-1])]
        
        pool = torch.stack([pool_fn(o).unsqueeze(0) for o in outputs])
        y_hat = enc.classification_layer(pool.squeeze(1))
        
        znorm = torch.norm(zis.float(), p=1, dim=1)
        zdist = torch.sum(torch.abs(zis[:,:-1] - zis[:,1:]), dim=1)
        omega = (lamb1 * znorm) + (lamb2 * zdist)
        cost = xent(y_hat, label.repeat(num_samples)) + omega
        
        for sample in range(num_samples):
            for idx,p in enumerate(gen.parameters()):
                grads[idx] += cost[sample] * all_grads[idx][sample] 
        
        for idx,p in enumerate(gen.parameters()):
            p.data = p.data - learning_rate * (grads[idx] / float(num_samples))
            
        loss = cost.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.detach().cpu().numpy()

    print('Epoch: {0}, Train Loss: {1:0.4f}'.format(e, total_loss))