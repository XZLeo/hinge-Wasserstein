from copy import deepcopy
import os
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import Resize, CenterCrop, Normalize, ToTensor
from torchvision.transforms.functional import hflip

from src.functionals import val2bin, leftright2normal

from logging import getLogger

"""Logger for printing."""
_LOG = getLogger(__name__)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
# STD = [1, 1, 1]

class SplitDataset(Dataset):
    '''
    load train/val/test dataset in the order of txt
    '''
    def __init__(self, dataset:str, mode: str, csv_path: Path, txt_path: Path, 
                 img_path: Path, flip_thresh:float, img_size: int = 224):
        self.img_size = img_size
        self.data = pd.read_csv(csv_path)
        self.split_img_paths = read_txt(txt_path)
        self.mode = mode
        self.transform = None
        self.img_path = img_path
        self.flip_thresh = flip_thresh
        if dataset not in ('toy', 'hlw'):
            raise Exception('wong dataset type, either toy or hlw') 
        self.dataset = dataset

    def __len__(self):
        return len(self.split_img_paths)

    def __getitem__(self, index):
        #load img
        img_name = self.split_img_paths[index]
        full_img_path = os.path.join(self.img_path, img_name)
        img = Image.open(full_img_path).convert('RGB')
        img = deepcopy(img)
        # Data augmentation
        crop_size = min(img.size)
        if self.mode == "Train":
            # randomly mirroring the image horizontally with 50% probability 
            # sampling a square crop (minimum side length 85% of the smallest image dimension)
            # resize to 224*224
            crop_size = crop_size * np.random.uniform(low=0.85, high=1, size=(1,))
            self.transform = T.Compose([
                CenterCrop(crop_size[0]), # disable random crop size
                Resize(self.img_size),
                ToTensor(),
                Normalize(MEAN, STD)
            ])
        elif self.mode == "Valid" or self.mode == "Test":
            # maximal_square_center_crop
            # resize to 224*224
            self.transform = T.Compose([ 
                CenterCrop(crop_size),
                Resize(self.img_size),
                ToTensor(),
                Normalize(MEAN, STD)
            ])
        else:
            raise "Unsupported data read type. Please use `Train` or `Valid` or `Test`"
        
        if self.mode == "Train":
            img, flip_flag = randomHorizontalFlip(img, self.flip_thresh)
        else:
            flip_flag = False
        
        img = self.transform(img)
                
        # label (l,r) ==> (theta, rho)
        data = deepcopy(self.data)
        row_idx = data[data["image_name"]== img_name].index.tolist()[0]
        row = data.iloc[row_idx]

        if self.dataset == 'toy':
            theta, rho = leftright2normal(float(row["left_y"]), float(row["left_x"]),
                            float(row["right_y"]), float(row["right_x"]), crop_size)
        else:
            # for testing toy dataset, choose the first line among two GND lines
            theta, rho = leftright2normal(float(row["left_y_1"]), float(row["left_x"]),
                                float(row["right_y_1"]), float(row["right_x"]), crop_size)
        
        if flip_flag:  # horizontal flip doesn't change rho
            theta = -theta
        return (img, (theta, rho))


class ClassificationDataset(SplitDataset):
    '''
    load train/val/test dataset in the order of txt, return theta_id and rho_id in 100 bins
    '''
    def __init__(self, mode: str, csv_path: Path, txt_path: Path,
                 img_path: Path, bin_edges: Dict, flip_thresh:float=0.5, img_size: int = 224):
        super().__init__(mode, csv_path, txt_path, img_path, flip_thresh, img_size)
        self.bin_edges = bin_edges

    def __getitem__(self, index):
        (img, (theta, rho)) = super().__getitem__(index)
        theta_id, rho_id = val2bin(theta, rho, self.bin_edges)
        return (img, (theta_id, rho_id), (theta, rho))

    def __len__(self):
        return super().__len__()


def read_txt(txt_path:Path):
    '''
    read train.txt, test.txt, val.txt
    return img_paths: a list of path string
    '''
    with open(txt_path,'r',encoding = 'UTF-8') as f:
        img_paths = f.readlines()
    img_paths = [c.strip() for c in img_paths]
    return img_paths


def randomHorizontalFlip(img:np.ndarray, thresh:float=0.5):
    '''
    return flag: True for flipped, false for not, useful for label adjustment
           img: transformed img
    '''
    if torch.rand(1) < thresh:
        img = hflip(img)
        flag = True
    else:
        flag = False    
    return img, flag


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in enumerate(self.batch_data):
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True) # list not dict

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)


class TwoLineDataset(Dataset):
    '''
    Compatible with a mixed length of GND, either 1 line or 2 lines, for loading 1 line or both lines in the toy dataset
    '''
    def __init__(self, mode: str, csv_path: Path, txt_path: Path, img_path: Path, flip_thresh:float, img_size: int = 224):
        self.img_size = img_size
        self.data = pd.read_csv(csv_path)
        self.split_img_paths = read_txt(txt_path)
        self.mode = mode
        self.transform = None
        self.img_path = img_path
        self.flip_thresh = flip_thresh

    def __len__(self):
        return len(self.split_img_paths)

    def __getitem__(self, index):
        #load img
        img_name = self.split_img_paths[index]
        full_img_path = os.path.join(self.img_path, img_name)
        img = Image.open(full_img_path).convert('RGB')
        img = deepcopy(img)
        # Data augmentation
        crop_size = min(img.size)
        if self.mode == "Train":
            # randomly mirroring the image horizontally with 50% probability 
            # sampling a square crop (minimum side length 85% of the smallest image dimension)
            # resize to 224*224
            crop_size = crop_size * np.random.uniform(low=0.85, high=1)
            self.transform = T.Compose([
                CenterCrop(crop_size), # disable random crop size
                Resize(self.img_size),
                ToTensor(),
                Normalize(MEAN, STD)
            ])
        elif self.mode == "Valid" or self.mode == "Test":
            # maximal_square_center_crop
            # resize to 224*224
            self.transform = T.Compose([ 
                CenterCrop(crop_size),
                Resize(self.img_size),
                ToTensor(),
                Normalize(MEAN, STD)
            ])
        else:
            raise "Unsupported data read type. Please use `Train` or `Valid` or `Test`"
        
        if self.mode == "Train":
            img, flip_flag = randomHorizontalFlip(img, self.flip_thresh)
        else:
            flip_flag = False
        
        img = self.transform(img)
                
        # label (l,r) ==> (theta, rho)
        data = deepcopy(self.data)
        row_idx = data[data["image_name"]== img_name].index.tolist()[0]
        row = data.iloc[row_idx]

        theta_1, rho_1 = leftright2normal(float(row["left_y_1"]), float(row["left_x"]),
                            float(row["right_y_1"]), float(row["right_x"]), crop_size)
        
        theta_1 = -theta_1 if flip_flag else theta_1  # horizontal flip doesn't change rho
            
        
        if float(row["left_y_2"]) == float('inf')\
            or float(row["right_y_2"]) == float('inf'):
               theta_2, rho_2 = float('inf'), float('inf')
        else:           
            theta_2, rho_2 = leftright2normal(float(row["left_y_2"]), float(row["left_x"]),
                                float(row["right_y_2"]), float(row["right_x"]), crop_size)
            theta_2 = -theta_2 if flip_flag else theta_2  # horizontal flip doesn't change rho
                
        return (img, (theta_1, rho_1), (theta_2, rho_2))


class TwoLineClassDataset(TwoLineDataset):
    '''
    load train/val/test dataset in the order of txt, return theta_id and rho_id in 100 bins
    '''
    def __init__(self, mode: str, csv_path: Path, txt_path: Path,
                 img_path: Path, bin_edges: Dict, flip_thresh:float=0.5, img_size: int = 224):
        super().__init__(mode, csv_path, txt_path, img_path, flip_thresh, img_size)
        self.bin_edges = bin_edges

    def __getitem__(self, index):
        img, (theta_1, rho_1), (theta_2, rho_2) = super().__getitem__(index)
        theta_1_id, rho_1_id = val2bin(theta_1, rho_1, self.bin_edges)
        if theta_2 == float('inf') or rho_2 == float('inf'):
            theta_2_id, rho_2_id = float('inf'), float('inf')
        else:
            theta_2_id, rho_2_id = val2bin(theta_2, rho_2, self.bin_edges)
        return (img, (theta_1_id, rho_1_id), (theta_2_id, rho_2_id), (theta_1, rho_1), (theta_2, rho_2))

    def __len__(self):
        return super().__len__()
