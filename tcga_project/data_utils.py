import os
import sys
import torch
#import accimage
from PIL import Image
from imageio import imread
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms, set_image_backend, get_image_backend

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
        
        #total = 0
        for im in imgs:
            path = img_dir + '/' + im
            image = self.loader(path)
            #image = imread(path)

            if self.transform is not None:
                #print(image.mode)
                image = self.transform(image)
                
            if image.shape[1] == 256 and image.shape[2] == 256:
                slide_tiles.append(image)
                #total += sys.getsizeof(image)
        
        slide = torch.stack(slide_tiles)
        #print(total/1e6)
        #slide = slide_tiles
        label = self.sample_labels[idx]
        sample = {'slide': slide, 'label': label}

        return sample