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
root_dir_all = '/n/mounted-data-drive/'


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

    def __init__(self, sample_annotations, root_dir, transform=None, loader=default_loader, magnification='5.0', 
                 batch_type='tile', tile_batch_size=800, all_cancers=False, cancer_type=None, return_jpg_to_sample=False, return_coords = False):
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
        self.cancer_type = cancer_type
        self.return_jpg_to_sample = return_jpg_to_sample
        self.return_coords = return_coords
        if all_cancers:
            self.img_dirs = [self.root_dir + self.cancer_type[idx] + '/' + sample_name + '.svs/' \
                            + sample_name + '_files/' + self.magnification for idx,sample_name in enumerate(self.sample_names)]
        else:
            self.img_dirs = [self.root_dir + sample_name + '.svs/' \
                             + sample_name + '_files/' + self.magnification for sample_name in self.sample_names]
        self.jpegs = [os.listdir(img_dir) for img_dir in self.img_dirs]
        self.all_jpegs = []
        self.all_labels = []
        self.jpg_to_sample = []
        self.coords = []
        self.tile_batch_size = tile_batch_size
        
        for idx,(im_dir,label,l) in enumerate(zip(self.img_dirs, self.sample_labels, self.jpegs)):
            sample_coords = []
            for jpeg in l:
                byte_jpeg = bytearray(jpeg.encode())
                self.all_jpegs.append(im_dir + '/' + jpeg)
                self.all_labels.append(label)
                self.jpg_to_sample.append(idx)
                x,y = byte_jpeg[:-5].split('_'.encode()) # 'X_Y.jpeg'
                x,y = int(x), int(y)
                sample_coords.append(torch.tensor([x,y]))
            self.coords.append(torch.stack(sample_coords))    
        self.all_coords = torch.cat(self.coords) 
        
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
            if self.return_jpg_to_sample and self.return_coords:
                return image, self.all_labels[idx], self.jpg_to_sample[idx], self.all_coords[idx]
            elif self.return_jpg_to_sample:
                return image, self.all_labels[idx],self.jpg_to_sample[idx]
            elif self.return_coords:
                return image, self.all_labels[idx], self.all_coords[idx]
            else:
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
    
       
def get_sample_annotations(sample_names, sample_idxs, df_raw, name_len, col_name):
    sample_annotations = {}
    sample_names = np.array(sample_names)
    for sample_name in sample_names[sample_idxs]:
        sample_annotations[sample_name] = df_raw.loc[sample_name[0:name_len], col_name]
    return sample_annotations


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
    
    col_name = 'MSI.int'
    msi_raw[col_name] = msi_label
    idx = int(np.floor(len(sample_names)*0.8))
    train = reorder[:idx]
    val = reorder[idx:]        
    sample_annotations_train = get_sample_annotations(sample_names, train, msi_raw, name_len, col_name)        
    sample_annotations_val = get_sample_annotations(sample_names, val, msi_raw, name_len, col_name)        
    return sample_annotations_train, sample_annotations_val        


def process_WGD_data(root_dir='/n/mounted-data-drive/', cancer_type='COAD', wgd_path='/home/amaro/MSI_prediction/tcga_project/misc/COAD_WGD_TABLE.xls', 
                     wgd_raw=None, split_in_two=False, print_stats=False):
    # note: flexible across cancer types, e.g., coad vars can be for brca
    if wgd_path is not None:
        wgd_raw = pd.read_excel(wgd_path,index_col='Sample')
    sample_name = wgd_raw.index[0]
    name_len = len(sample_name)
    coad_full_name = os.listdir(root_dir + cancer_type)
    coad_img = np.array([v[0:name_len] for v in coad_full_name])
    coad_both = np.intersect1d(coad_img, wgd_raw.index)        
        
    if print_stats:
        if cancer_type[-1:] == 'x':
            num_labels = np.sum(wgd_raw['Type'].isin([cancer_type.split('_')[0]]))
        else:
            num_labels = np.sum(wgd_raw['Type'].isin([cancer_type]))
        print('{0:<8}  Num Images: {1:>5,d}  Num Labels: {2:>5,d}  Overlap: {3:>5,d}'.format(cancer_type, len(coad_img),
                                                                                             num_labels, len(coad_both)))    
    sample_names = []
    for sample in coad_both:
        key = np.argwhere(coad_img == sample).squeeze()
        if key.size != 0:
            sample_names.append(coad_full_name[key][:-4])
    reorder = np.random.permutation(len(sample_names))
    
    col_name = 'Genome_doublings'
    if split_in_two:
        idx1 = int(np.floor(len(sample_names)*0.4))
        idx2 = int(np.floor(len(sample_names)*0.1))
        train1 = reorder[:idx1]
        val1 = reorder[idx1:(idx1+idx2)]
        train2 = reorder[(idx1+idx2):(idx1+idx2+idx1)]
        val2 = reorder[(idx1+idx2+idx1):]                
        sample_annotations_train1 = get_sample_annotations(sample_names, train1, wgd_raw, name_len, col_name)                
        sample_annotations_val1 = get_sample_annotations(sample_names, val1, wgd_raw, name_len, col_name)               
        sample_annotations_train2 = get_sample_annotations(sample_names, train2, wgd_raw, name_len, col_name)                
        sample_annotations_val2 = get_sample_annotations(sample_names, val2, wgd_raw, name_len, col_name)        
        return sample_annotations_train1, sample_annotations_val1, sample_annotations_train2, sample_annotations_val2
    else:
        idx = int(np.floor(len(sample_names)*0.8))
        train = reorder[:idx]
        val = reorder[idx:]        
        sample_annotations_train = get_sample_annotations(sample_names, train, wgd_raw, name_len, col_name)        
        sample_annotations_val = get_sample_annotations(sample_names, val, wgd_raw, name_len, col_name)
        return sample_annotations_train, sample_annotations_val


def load_COAD_train_val_sa_pickle(pickle_file='/n/tcga_models/resnet18_WGD_10x_sa.pkl', 
                                  return_all_cancers=False, split_in_two=False):
    empty_sample = 'TCGA-A6-2675-01Z-00-DX1.d37847d6-c17f-44b9-b90a-84cd1946c8ab'
    with open(pickle_file, 'rb') as f: 
        if return_all_cancers:
            if split_in_two:
                batch_all, sa_trains1, sa_vals1, sa_trains2, sa_vals2 = pickle.load(f)
                return batch_all, sa_trains1, sa_vals1, sa_trains2, sa_vals2
            else:
                batch_all, sa_trains1, sa_vals1 = pickle.load(f)
                return batch_all, sa_trains1, sa_vals1
        else:
            sa_train, sa_val = pickle.load(f)
            if empty_sample in list(sa_train.keys()):
                del sa_train[empty_sample]
            elif empty_sample in list(sa_val.keys()):
                del sa_val[empty_sample]
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
        
        for idx,(im_dir,label,l) in enumerate(zip(self.img_dirs, self.sample_labels, self.jpegs)):
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
        return image, self.all_labels[idx], self.coords[idx], self.jpg_to_sample[idx]
    
    
    
    
    
    
    
    
    
    
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
    
    
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets, return_jpg_to_sample=False):
        self.datasets = datasets
        self.return_jpg_to_sample = return_jpg_to_sample
        
    def __len__(self):
        return min(len(d) for d in self.datasets)
    
    def __getitem__(self, i):
        if self.return_jpg_to_sample:
            return torch.stack([d[i][0] for d in self.datasets]), \
        torch.cat([torch.tensor(d[i][1]).view(-1) for d in self.datasets]), \
        torch.stack([torch.tensor([c, d.jpg_to_sample[i]]) for c,d in enumerate(self.datasets)])
        else:
            return torch.stack([d[i][0] for d in self.datasets]), torch.cat([torch.tensor(d[i][1]).view(-1) \
                                                                             for d in self.datasets])
        
        
class MergedDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets, return_jpg_to_sample=False):
        self.datasets = datasets
        self.all_jpgs = [d.all_jpegs for d in datasets]
        self.return_jpg_to_sample = return_jpg_to_sample
        self.jpg_to_sample = [d.jpg_to_sample for d in datasets]
        self.labels = [d.all_labels for d in datasets]
        self.transform = datasets[0].transform
        self.loader = datasets[0].loader
    def make_image(self,image):
        image = self.loader(image)
        image = self.transform(image)
        if image.shape[1] < 256 or image.shape[2] < 256:
                image = pad_tensor_up_to(image,256,256,channels_last=False)
        return image
    def __len__(self):
        return min(len(d) for d in self.datasets)
    
    def __getitem__(self, i):
        if self.return_jpg_to_sample:
            return torch.stack([self.make_image(jpegs[i]) for jpegs in self.all_jpgs]), \
        torch.cat([torch.tensor(labels[i]).view(-1) for labels in self.labels]), \
        torch.stack([torch.tensor([c, jpg_to_sample[i]]) for c,jpg_to_sample in enumerate(self.jpg_to_sample)])
        else:
            return torch.stack([self.make_image(jpegs[i]) for jpegs in self.all_jpgs]), \
        torch.cat([torch.tensor(labels[i]).view(-1) for labels in self.labels])