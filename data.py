import re
from pathlib import Path
from random import random

import torch
from more_itertools import flatten
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

from util import dose2locs, loc2dose, identity
from PIL.Image import BILINEAR

def transforms1(image_size, w=3, zoom=1.1):
    return [
        transforms.Resize(image_size),
        transforms.RandomAffine(w, (.01*w, .01*w), (1, 1), w, BILINEAR),
        transforms.Resize(int(image_size*zoom)), 
        transforms.CenterCrop(image_size)   
    ]


class DoseCurveDataset(Dataset):
    def __init__(self, folder, image_size, chans=[0,1,2,3,4], train=True, norm_f=None,
                 w=None, doses="all", label=False):

        if doses == "all":
            doses = dose2locs.keys()
        w = w or (3 if train else 0)
        
        def paths(folder, doses):
            not_52 = re.compile('/[^(52)]')
            assays = flatten(dose2locs[dose] for dose in doses)
            gen = flatten((Path(f'{folder}').glob(
                f'**/*{assay}*.pt')) for assay in assays)
            return [p for p in gen if not_52.search(str(p))]

        self.dose2id = {k: i for i, k in enumerate(doses)}
        self.f = d8 if train else identity
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.label = label
        self.norm_f = norm_f or identity

        self.paths = paths(folder, doses)
        assert len(self.paths) > 0, f'No images were found in {folder} for training'

        #convert_image_fn = convert_transparent_to_rgb if not transparent else convert_rgb_to_transparent
        self.chans = chans 

        self.transform = transforms.Compose(transforms1(image_size, w))
        

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = self.norm_f(torch.load(path))
        
        img = img[self.chans]
        
        if self.label:
            label = self.dose2id[loc2dose[str(path).split()[-2]]]
            return self.transform(self.f(img/255)), label
        return self.transform(self.f(img/255))


class MSNorm:  
    def __init__(self, norm_path):
        self.mean, self.std = torch.load(norm_path, map_location='cpu')
        
    def __call__(self, img):
        return (img - self.mean) / self.std

    def invert(self, img):
        return img * self.std + self.mean

    
def denorm_f(ms, device):
    mean, std = map(lambda x: torch.tensor(x,  device=device)[None, :, None, None], ms)
    return lambda x: (x*std + mean).cpu()

def d8(img):
    img = torch.rot90(img, int(random()*4), dims=(1,2))
    if random()>.5:
        img = torch.flip(img, dims=(2,))
    return img
