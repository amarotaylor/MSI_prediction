import pandas as pd
import numpy as np
import torch
import os
from scipy.io import loadmat
from imageio import imread
import matplotlib.pyplot as plt

COAD_IMG_DIR = '/n/data_labeled_histopathology_images/COAD/CRCHistoPhenotypes_2016_04_28/Classification/'


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
                slide_tiles.append(torch.tensor(bmp_im[y-13:y+14,x-13:x+14,:],dtype=torch.float32))
                slide_tile_class.append(cell_label)
                slide_locs.append((x,y))
    slide = torch.stack(slide_tiles)
    return slide,slide_tile_class,slide_locs
