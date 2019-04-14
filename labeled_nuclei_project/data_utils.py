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
    Read a slide and return 3d array of tiles, list 
    of classes for each tile, and tile coordinates
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


def make_image_from_slide(valid,idx):
    fig,ax = plt.subplots(1,1)
    image = np.zeros([500,500,3])
    locs = valid.cell_locs[idx]
    slide = valid.data[idx].squeeze()
    for i,tile in enumerate(slide):    
        image[locs[i][1]-13:locs[i][1]+14,locs[i][0]-13:locs[i][0]+14] = tile.numpy().reshape(27,27,3)#*255.0*((a_np[i]-a_min)/(a_max - a_min))

    ax.imshow(image)
    fig.set_dpi(150)

    
def frame_image(img, frame_width):
    b = frame_width # border size in pixel
    ny, nx = img.shape[0], img.shape[1] # resolution / number of pixels in x and y
    if img.ndim == 3: # rgb or rgba array
        framed_img = np.zeros((ny, nx, img.shape[2]))
        framed_img[:,:,0]+=1
    elif img.ndim == 2: # grayscale image
        framed_img = np.zeros((b+ny+b, b+nx+b))
    framed_img[b:-b, b:-b] = img[b:-b, b:-b]
    return framed_img


def draw_image_with_rationale(idx,valid,gen):
    fig,ax = plt.subplots(1,1)
    image = np.zeros([500,500,3])
    locs = valid.cell_locs[idx]
    slide = valid.data[idx].squeeze().view(-1,3,27,27).cuda()
    rationale = gen(slide)
    keep = torch.argmax(rationale,2).squeeze()
    slide = slide.cpu().view(-1,27,27,3).numpy()
    
    for i,tile in enumerate(slide):
        if keep[i] == 0:
            image[locs[i][1]-13:locs[i][1]+14,locs[i][0]-13:locs[i][0]+14] = tile#*255.0*((a_np[i]-a_min)/(a_max - a_min))
            
    for i,tile in enumerate(slide):
        if keep[i] == 1:
            tile = frame_image(tile,3)
            image[locs[i][1]-13:locs[i][1]+14,locs[i][0]-13:locs[i][0]+14] = tile#*255.0*((a_np[i]-a_min)/(a_max - a_min))

    ax.imshow(image)
    fig.set_dpi(150)
    
    
class COAD_dataset(Dataset):
    '''
    Torch dataset for colorectal images.
    Slide level labels for prediction : self.labels
    Slide examples : self.data
    Slide cell level labels : self.cell_labels
    Slide cell level x,y coords : self.cell_locs
    '''
    def __init__(self, pickle_file, return_cell_positions = False):
        with open(pickle_file, 'rb') as f: 
            all_slides, all_slide_tile_classes, all_slide_locs, all_slide_labels = pickle.load(f)
        self.labels = all_slide_labels
        self.data = all_slides
        self.cell_labels = all_slide_tile_classes
        self.cell_locs = all_slide_locs
        self.return_cell_positions = return_cell_positions
        if self.return_cell_positions:
            self.neighborhoods = list()
            for slide in range(len(self.data)):
                neighbors = list()
                x_y_arr = np.array(self.cell_locs[slide])
                for idx,xy in enumerate(x_y_arr):
                    neighbors.append(np.argsort(np.sum((xy - x_y_arr)**2,1))[1:5])
                self.neighborhoods.append(np.stack(neighbors))
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.return_cell_positions:
            return self.data[idx].view(-1,3,27,27),self.labels[idx],np.array(self.neighborhoods[idx])
        else:
            return self.data[idx].view(-1,3,27,27),self.labels[idx]