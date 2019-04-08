import pandas as pd
import numpy as np
import torch
import os
from scipy.io import loadmat
from imageio import imread
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import pickle


COAD_IMG_DIR = '/n/data_labeled_histopathology_images/COAD/CRCHistoPhenotypes_2016_04_28/Classification/'
COAD_TRAIN = '/n/data_labeled_histopathology_images/COAD/train.pkl'
COAD_VALID = '/n/data_labeled_histopathology_images/COAD/valid.pkl'
COAD_DEV = '/n/data_labeled_histopathology_images/COAD/dev.pkl'
COAD_TEST = '/n/data_labeled_histopathology_images/COAD/test.pkl'
COAD_DATASET = '/n/data_labeled_histopathology_images/COAD/dataset.pkl'


def process_slide(current_img,imgs,i):
    bmp = current_img+'/'+imgs[i]+'.bmp'
    epi_mat = current_img+'/'+imgs[i]+'_epithelial.mat'
    fibro_mat = current_img+'/'+imgs[i]+'_fibroblast.mat'
    inf_mat = current_img+'/'+imgs[i]+'_inflammatory.mat'
    other_mat = current_img+'/'+imgs[i]+'_others.mat'
    bmp_im = imread(bmp)
    epis = loadmat(epi_mat)
    if len(epis['detection'])==0:
        slide_level_label = 0
    else:
        slide_level_label = 1
    return slide_level_label,epi_mat,fibro_mat,inf_mat,other_mat,bmp_im

def read_slide(epi_mat,fibro_mat,inf_mat,other_mat,bmp_im):
    '''
    read a slide and return 3d array of tiles, list 
    of classes for each tile and tile coordinates
    '''
    slide_tiles = []
    slide_tile_class = []
    slide_locs = []
    classes = ['epi','fibro','inf','other']
    mat_files = [epi_mat,fibro_mat,inf_mat,other_mat]

    for matfile,cell_label in zip(mat_files,classes):
        cell_dict = loadmat(matfile)['detection']
        for cell in range(len(cell_dict)):
            [x,y] = cell_dict[cell].astype(int)
            if x<13 or (x+14)>500 or y<13 or (y+14)>500:
                pass
            else:
                slide_tiles.append(torch.tensor(bmp_im[y-13:y+14,x-13:x+14,:],dtype=torch.float32)/255.0)
                slide_tile_class.append(cell_label)
                slide_locs.append((x,y))
    slide = torch.stack(slide_tiles)
    return slide,slide_tile_class,slide_locs



class COAD_dataset(Dataset):
    '''
    Torch dataset for colorectal images.
    Slide level labels for prediction : self.labels
    Slide examples : self.data
    Slide cell level labels : self.cell_labels
    Slide cell level x,y coords : self.cell_locs
    '''
    def __init__(self,pickle_file):
        with open(pickle_file, 'rb') as f: 
            all_slides, all_slide_tile_classes, all_slide_locs,all_slide_labels = pickle.load(f)
        self.labels = all_slide_labels
        self.data = all_slides
        self.cell_labels = all_slide_tile_classes
        self.cell_locs = all_slide_locs
        
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.data[idx].view(-1,3,27,27),self.labels[idx]
