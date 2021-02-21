from statistics import mean 

import torch
from matplotlib import pyplot as plt
from more_itertools import take
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.utils import make_grid
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dose2locs = {
    0.0: ['B10','C5','C9','C10','E2','E7','F7','G7'],
    0.1: ['C2','C11','D6','D8','D10','E4','E8','F3','G9'],
    0.3: ['B5', 'D7', 'D9','D11','E6','E11','F6','F10','F11'],
    1.0: ['B4','B7','C4','E5','E9','F4','F8','G5','G6'],
    3.0: ['B6','D2','D3','D4','E3','F9','G3','G4'],
    30.0: ['B8','B9','C3','C6','C7','C8','E10','F2','G8']
}
loc2dose = dict()
for k, vs in dose2locs.items():
    for v in vs:
        loc2dose[v] = k

def show(g):
    plt.figure(figsize=(40,40)) 
    plt.imshow(g[0,:3].detach().cpu().permute(1,2,0))

def show2(batch, cols=4):
    plt.figure(figsize=(40,40)) 
    plt.imshow((make_grid(batch, cols)[:3]).permute(1,2,0))

class Normalization(nn.Module):
    def __init__(self, mean=torch.zeros(3), std=torch.ones(3)):
        super(Normalization, self).__init__()
        self.mean = nn.Parameter(mean.view(-1, 1, 1), requires_grad=False)
        self.std = nn.Parameter(std.view(-1, 1, 1), requires_grad=False)

    def forward(self, img):
        return (img - self.mean) / self.std


class Aggregator:
    def __init__(self, model, style_layers, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.style_layers = style_layers
        self.model = transferrer(model, style_layers)
        self.total = []
        self.total2 = []
        self.n = 0

    @torch.no_grad()
    def vecs(self, ims):
        x = ims.to(device)
        r = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.style_layers:
                r.append(gram_matrices(x))
        return r
        
    @staticmethod
    def update(total, by, p=1):
        if len(total) == 0:
            total = [(x**p).sum(0) for x in by]
        else:
            for i in range(len(by)):
                total[i] += (by[i]**p).sum(0)
        return total

    def _step(self, by):
        self.total = self.update(self.total, by)
        self.total2 = self.update(self.total2, by, 2)
        self.n += by[0].shape[0]

    def step(self, ims):
        by = self.vecs(ims)
        self._step(by)

    def get_mss(self):
        mus = [x/self.n for x in self.total]
        stds = [torch.sqrt((snd/self.n) - fst**2) for fst, snd in zip(mus, self.total2)]
        return list(zip(mus, stds))
        
    
def assess_transfer(transfer, classifier, data_path, dose_c, dose_s,
                    batch_size=4, image_size=512, samples=20):
    # either take some images or do aggregate
    data_c, data_s = map(lambda x: DataLoader(ImageDataset(data_path, image_size, train=False, doses=[x])),
                         [dose_c, dose_s])

    n = samples/batch_size
    for b_c, b_s in take(n, zip(iter(data_c), iter(data_s))):
        print(F.softmax(classifier(transfer(b_c, b_s)), 1))
    
    
def evaluate_median(loader):
    losses = []
    for batch_content, _ in tqdm(loader):
        medians = batch_content.median(2).values.median(2).values[:,:,None,None]
        losses.append(F.l1_loss(batch_content, medians).item())
    return mean(losses)


def transfer_vis(transfer, batch_content, batch_style, ms):
    mean, std = map(lambda x: x[None, :, None, None], ms)
    output = transfer(batch_content, batch_style)
    f = lambda x: x*std + mean
    
class identity(object):
    @staticmethod
    def __call__(tensor):
        return tensor
