import os
import sys
import torch
import accimage
import numpy as np
import pandas as pd
from PIL import Image
from imageio import imread
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms, set_image_backend, get_image_backend
import torch.nn.functional as F
import pickle
import train_utils
import data_utils
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
set_image_backend('accimage')
import data_utils
import train_utils
root_dir_coad = '/n/mounted-data-drive/COAD/'



sa_train, sa_val = data_utils.load_COAD_train_val_sa_pickle()
root_dir = '/n/mounted-data-drive/COAD/'
magnification = '10.0'
batch_type = 'slide'

train_transform = train_utils.transform_train
val_transform = train_utils.transform_validation



train_set = data_utils.TCGADataset_tiled_slides(sa_train, root_dir, transform=train_transform, magnification=magnification)
train_loader = DataLoader(train_set, batch_size=256, pin_memory=True, num_workers=32)


val_set = data_utils.TCGADataset_tiled_slides(sa_val, root_dir, transform=val_transform, magnification=magnification)
val_loader = DataLoader(val_set, batch_size=256, pin_memory=True, num_workers=32)


state_dict_file = '/n/tcga_models/resnet18_WGD_10x.pt'
device = torch.device('cuda', 0)
output_shape = 1


resnet = models.resnet18(pretrained=False)
resnet.fc = nn.Linear(2048, output_shape, bias=True)
saved_state = torch.load(state_dict_file, map_location=lambda storage, loc: storage)
resnet.load_state_dict(saved_state)
resnet.fc = nn.Linear(2048, 2048, bias=False)
resnet.fc.weight.data = torch.eye(2048)
resnet.cuda(device=device)
for param in resnet.parameters():
    param.requires_grad = False
    
    
    
def pool_fn(x):
    #v,a = torch.max(x,0)
    v = torch.mean(x,0)
    return v


slide_level_classification_layer = nn.Linear(2048,1)
slide_level_classification_layer.cuda()



e = 0
learning_rate = 1e-4
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(slide_level_classification_layer.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, min_lr=1e-8)




best_loss = 1e8
best_acc = 0.0

for e in range(200):
    train_utils.tcga_tiled_slides_training_loop(e, train_loader, resnet, 
                                    slide_level_classification_layer, criterion, 
                                 optimizer, pool_fn,device=device,train_set=train_set)

    loss, acc = train_utils.tcga_tiled_slides_validation_loop(e, val_loader, resnet, 
                                    slide_level_classification_layer, criterion, 
                                 scheduler, pool_fn,device=device,val_set=val_set)

    if best_loss > loss or best_acc < acc:
        print('Wrote model')
        torch.save(slide_level_classification_layer.state_dict(),'linear_classifier.pt')






