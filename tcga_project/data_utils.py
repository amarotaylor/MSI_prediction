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

set_image_backend('accimage')
max_tiles = 100
root_dir_coad = '/n/mounted-data-drive/COAD/'


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

        
def accimage_loader(path):
    try:
        return accimage.Image(path)
    except IOError:
        # potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

    
def default_loader(path):
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
    
    
class TCGADataset(Dataset):
    """TCGA dataset."""

    def __init__(self, sample_annotations, root_dir, transform=None, loader=default_loader):
        """
        Args:
            sample_annot (dict): dictionary of sample names and their respective labels.
            root_dir (string): directory containing all of the samples and their respective images.
            transform (callable, optional): optional transform to be applied on the images of a sample.
        """
        self.sample_names = list(sample_annotations.keys())
        self.sample_labels = list(sample_annotations.values())
        self.root_dir = root_dir
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):
        slide_tiles = []
        img_dir = self.root_dir + self.sample_names[idx] + '.svs/' + self.sample_names[idx] + '_files/5.0'
        imgs = os.listdir(img_dir)
        num_tiles = len(imgs)
        imgs = np.array(imgs)[torch.randperm(num_tiles)[:max_tiles]]
        
        for im in imgs:
            path = img_dir + '/' + im
            image = self.loader(path)

            if self.transform is not None:
                image = self.transform(image)
#                image = image#/255.0
            if image.shape[1] == 256 and image.shape[2] == 256:
                slide_tiles.append(image)
        
        slide = torch.stack(slide_tiles)
        label = self.sample_labels[idx]

        return slide, label
    
    
def pad_tensor_up_to(x,H,W,channels_last=True):
    if channels_last==False:
        p1d = (0,W - x.shape[2],0,H - x.shape[1],0,0)
    
    else:
        p1d = (0,0,0,W - x.shape[1],0,H - x.shape[0])
    
    return F.pad(x, p1d, "constant", 0)     
    
    
class TCGADataset_tiles(Dataset):
    """TCGA dataset. This dataset treads a middle ground enabling both tile and slide level learning.
    In tile level mode it returns a bag of tiles each with their own label. 
    In slide level mode it returns slides in chunks of up to tile_batch_size tiles in order.
    """

    def __init__(self, sample_annotations, root_dir, transform=None, loader=default_loader, 
                 magnification='5.0', batch_type='tile', tile_batch_size=800):
        """
        Args:
            sample_annot (dict): dictionary of sample names and their respective labels.
            root_dir (string): directory containing all of the samples and their respective images.
            transform (callable, optional): optional transform to be applied on the images of a sample.
        """
        self.sample_names = list(sample_annotations.keys())
        self.sample_labels = list(sample_annotations.values())
        self.root_dir = root_dir
        self.transform = transform
        self.loader = loader
        self.magnification = magnification
        self.batch_type = batch_type
        self.img_dirs = [self.root_dir + sample_name + '.svs/' \
                         + sample_name + '_files/' + self.magnification for sample_name in self.sample_names]
        self.jpegs = [os.listdir(img_dir) for img_dir in self.img_dirs]
        self.all_jpegs = []
        self.all_labels = []
        self.jpg_to_sample = []
        self.coords = []
        self.tile_batch_size = tile_batch_size
        for idx,(im_dir,label,l) in enumerate(zip(self.img_dirs,self.sample_labels,self.jpegs)):
            sample_coords = []
            for jpeg in l:
                self.all_jpegs.append(im_dir+'/'+jpeg)
                self.all_labels.append(label)
                self.jpg_to_sample.append(idx)
                x,y = jpeg[:-5].split('_') # 'X_Y.jpeg'
                x,y = int(x), int(y)
                sample_coords.append(torch.tensor([x,y]))
            self.coords.append(torch.stack(sample_coords))
                 
    def __len__(self):
        if self.batch_type == 'tile':
            return len(self.all_jpegs)
        elif self.batch_type == 'slide':
            return len(self.jpegs)

    def __getitem__(self, idx):
        if self.batch_type == 'tile':
            image = self.loader(self.all_jpegs[idx])
            if self.transform is not None:
                image = self.transform(image)
            if image.shape[1] < 256 or image.shape[2] < 256:
                image = pad_tensor_up_to(image,256,256,channels_last=False)
            return image, self.all_labels[idx]
        elif self.batch_type == 'slide':
            slide_tiles = []
            tiles_batch = []
            for tile_num,im in enumerate(self.jpegs[idx]):
                path = self.img_dirs[idx] + '/' + im
                image = self.loader(path)
                if self.transform is not None:
                    image = self.transform(image)
                if image.shape[1] < 256 or image.shape[2] < 256:
                    image = pad_tensor_up_to(image,256,256,channels_last=False)
                tiles_batch.append(image)
                
                if (tile_num+1) % self.tile_batch_size == 0 :
                    tiles_batch = torch.stack(tiles_batch)
                    slide_tiles.append(tiles_batch)
                    tiles_batch = []
            # grab last batch
            tiles_batch = torch.stack(tiles_batch)
            slide_tiles.append(tiles_batch)
            if len(slide_tiles) > 1 and len(tiles_batch) < self.tile_batch_size:
                # drop last batch if smaller than tile batch size
                # only occurs in slides with more than tile batch size tiles
                slide_tiles = slide_tiles[:-1]
            if len(slide_tiles)>1:
                slide = torch.stack(slide_tiles)
            else:
                slide = tiles_batch
                
            label = self.sample_labels[idx]
            coords = self.coords[idx]
            return slide, label, coords
    
       
def process_MSI_data():
    root_dir = '/n/mounted-data-drive/COAD/'
    msi_path = 'COAD_MSI_CLASS.csv'
    msi_raw = pd.read_csv(msi_path,index_col=0)    
    msi_raw['barcode'] = [s.replace('.','-') for s in list(msi_raw.index)]
    missing_gels = np.argwhere(msi_raw[['MSI.gel']].isnull().values)[:,0]
    msi_label = msi_raw['MSI.gel'].values
    
    for i in missing_gels:
        if msi_label[i]:
            msi_label[i]='MSI_L'
        else:
            msi_label[i]='MSS'
    b, c = np.unique(msi_label, return_inverse=True)
    msi_label = 2 - c     
    sample_name = msi_raw.iloc[-1]['barcode']
    name_len = len(sample_name)
    coad_full_name = os.listdir(root_dir)
    coad_img = np.array([v[0:name_len] for v in coad_full_name])
    coad_both = np.intersect1d(coad_img, msi_raw.barcode)
    sample_names = []
    
    for sample in coad_both:
        key = np.argwhere(coad_img == sample).squeeze()
        if key.size != 0:
            sample_names.append(coad_full_name[key][:-4])
    msi_raw.set_index('barcode', inplace=True)
    reorder = np.random.permutation(len(sample_names))
    train = reorder[:int(np.floor(len(sample_names)*0.8))]
    val = reorder[int(np.floor(len(sample_names)*0.8)):]
    
    sample_annotations = {}
    sample_names = np.array(sample_names)
    msi_raw['MSI.int'] = msi_label
    for sample_name in sample_names[train]:
        sample_annotations[sample_name] = msi_raw.loc[sample_name[0:name_len], 'MSI.int']
    sample_annotations_train = sample_annotations
    
    sample_annotations = {}
    sample_names = np.array(sample_names)
    msi_raw['MSI.int'] = msi_label
    for sample_name in sample_names[val]:
        sample_annotations[sample_name] = msi_raw.loc[sample_name[0:name_len], 'MSI.int']
    sample_annotations_val = sample_annotations
    
    return sample_annotations_train, sample_annotations_val
    
    
def process_WGD_data(root_dir='/n/mounted-data-drive/', cancer_type='COAD', wgd_path='COAD_WGD_TABLE.xls', wgd_raw=None):
    if wgd_path is not None:
        wgd_raw = pd.read_excel(wgd_path)
    sample_name = wgd_raw.index[0]
    name_len = len(sample_name)
    coad_full_name = os.listdir(root_dir + cancer_type)
    coad_img = np.array([v[0:name_len] for v in coad_full_name])
    coad_both = np.intersect1d(coad_img, wgd_raw.index)
    sample_names = []
    
    for sample in coad_both:
        key = np.argwhere(coad_img == sample).squeeze()
        if key.size != 0:
            sample_names.append(coad_full_name[key][:-4])
    reorder = np.random.permutation(len(sample_names))
    idx = int(np.floor(len(sample_names)*0.8))
    train = reorder[:idx]
    val = reorder[idx:]

    sample_annotations = {}
    sample_names = np.array(sample_names)
    for sample_name in sample_names[train]:
        sample_annotations[sample_name] = wgd_raw.loc[sample_name[0:name_len], 'Genome_doublings']
    sample_annotations_train = sample_annotations

    sample_annotations = {}
    sample_names = np.array(sample_names)
    for sample_name in sample_names[val]:
        sample_annotations[sample_name] = wgd_raw.loc[sample_name[0:name_len], 'Genome_doublings']
    sample_annotations_val = sample_annotations

    return sample_annotations_train, sample_annotations_val


def load_COAD_train_val_sa_pickle(pickle_file='/n/tcga_models/resnet18_WGD_10x_sa.pkl', return_all_cancers=False):
    with open(pickle_file, 'rb') as f: 
        if return_all_cancers:
            batch_all, sa_trains, sa_vals = pickle.load(f)
            return batch_all, sa_trains, sa_vals
        else:
            sa_train, sa_val = pickle.load(f)
            del sa_train['TCGA-A6-2675-01Z-00-DX1.d37847d6-c17f-44b9-b90a-84cd1946c8ab']
            return sa_train, sa_val
    
    
    
    
    
    
    
    
    
    
    
    
class TCGADataset_tiled_slides(Dataset):
    """
    TCGA slide dataset. Each slide is linked to its tiles via a label.
    This dataset returns a continuous stream of ordered tiles we then split those into update
    steps stochastically using random sampling
    """
    def __init__(self, sample_annotations, root_dir, transform=None, loader=default_loader, magnification='5.0'):
        """
        Args:
            sample_annot (dict): dictionary of sample names and their respective labels.
            root_dir (string): directory containing all of the samples and their respective images.
            transform (callable, optional): optional transform to be applied on the images of a sample.
            loader specifies image backend: use accimage
            magnification: tile magnification
        """
        self.sample_names = list(sample_annotations.keys())
        self.sample_labels = list(sample_annotations.values())
        
        self.root_dir = root_dir
        self.transform = transform
        self.loader = loader
        self.magnification = magnification
        self.img_dirs = [self.root_dir + sample_name + '.svs/' \
                         + sample_name + '_files/' + self.magnification for sample_name in self.sample_names]
        self.jpegs = [os.listdir(img_dir) for img_dir in self.img_dirs]
        self.all_jpegs = []
        self.all_labels = []
        self.jpg_to_sample = []
        self.coords = []
        
        for idx,(im_dir,label,l) in enumerate(zip(self.img_dirs,self.sample_labels,self.jpegs)):
            sample_coords = []
            for jpeg in l:
                # build tile dataset
                self.all_jpegs.append(im_dir+'/'+jpeg)
                # label for each tile
                self.all_labels.append(label)
                # tracks slide membership at a tile level
                self.jpg_to_sample.append(idx)
                # store tile coordinates
                x,y = jpeg[:-5].split('_') # 'X_Y.jpeg'
                x,y = int(x), int(y)
                self.coords.append(torch.tensor([x,y]))
            
    def __len__(self):
        return len(self.all_jpegs)
        
    def __getitem__(self, idx):
        image = self.loader(self.all_jpegs[idx])
        if self.transform is not None:
            image = self.transform(image)
        if image.shape[1] < 256 or image.shape[2] < 256:
            image = pad_tensor_up_to(image,256,256,channels_last=False)
        return image, self.all_labels[idx], self.coords[idx],self.jpg_to_sample[idx]
    
    
    
    
    
    
    
    
    
    
class TCGA_random_tiles_sampler(Dataset):
    """This data set samples tile_batch_size tiles from each slide and is run such that each
       slide is a single batch. 
     """

    def __init__(self, sample_annotations, root_dir, transform=None, loader=default_loader, 
                 magnification='5.0', tile_batch_size = 256):
        """
        Args:
            sample_annot (dict): dictionary of sample names and their respective labels.
            root_dir (string): directory containing all of the samples and their respective images.
            transform (callable, optional): optional transform to be applied on the images of a sample.
        """
        self.sample_names = list(sample_annotations.keys())
        self.sample_labels = list(sample_annotations.values())
        self.root_dir = root_dir
        self.transform = transform
        self.loader = loader
        self.magnification = magnification
        self.img_dirs = [self.root_dir + sample_name + '.svs/' \
                         + sample_name + '_files/' + self.magnification for sample_name in self.sample_names]
        self.jpegs = [os.listdir(img_dir) for img_dir in self.img_dirs]
        self.all_jpegs = []
        self.all_labels = []
        self.jpg_to_sample = []
        self.coords = []
        self.tile_batch_size = tile_batch_size
        for idx,(im_dir,label,l) in enumerate(zip(self.img_dirs,self.sample_labels,self.jpegs)):
            sample_coords = []
            for jpeg in l:
                self.all_jpegs.append(im_dir+'/'+jpeg)
                self.all_labels.append(label)
                self.jpg_to_sample.append(idx)
                x,y = jpeg[:-5].split('_') # 'X_Y.jpeg'
                x,y = int(x), int(y)
                sample_coords.append(torch.tensor([x,y]))
            self.coords.append(torch.stack(sample_coords))
                
            
    def __len__(self):
        ''' number of slides: jpegs is a list of lists '''
        return len(self.jpegs)

    def __getitem__(self, idx):
        slide_tiles = []
        tiles_batch = []
        perm = torch.randperm(len(self.jpegs[idx]))
        
        if len(self.jpegs[idx]) > self.tile_batch_size:
            idxs = perm[:self.tile_batch_size]
        else: 
            idxs = range(len(self.jpegs[idx]))
            
        for tile_num in idxs:
            im = self.jpegs[idx][tile_num]
            path = self.img_dirs[idx] + '/' + im
            image = self.loader(path)
            
            if self.transform is not None:
                image = self.transform(image)
            if image.shape[1] < 256 or image.shape[2] < 256:
                image = pad_tensor_up_to(image,256,256,channels_last=False)
            tiles_batch.append(image)

        # create batch of random tiles
        slide = torch.stack(tiles_batch)

        label = self.sample_labels[idx]
        coords = torch.stack([self.coords[idx][i] for i in idxs])
        return slide, label, coords