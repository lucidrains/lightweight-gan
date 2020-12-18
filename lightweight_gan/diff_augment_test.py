import os
import tempfile
from pathlib import Path
from shutil import copyfile

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from lightweight_gan.lightweight_gan import AugWrapper, ImageDataset


assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


@torch.no_grad()
def DiffAugmentTest(image_size = 256, data = './data/0.jpg', types = [], batch_size = 10, rank = 0, nrow = 5):
    model = DummyModel()
    aug_wrapper = AugWrapper(model, image_size)

    with tempfile.TemporaryDirectory() as directory:
        file = Path(data)

        if os.path.exists(file):
            file_name, ext = os.path.splitext(data)

            for i in range(batch_size):
                tmp_file_name = str(i) + ext
                copyfile(file, os.path.join(directory, tmp_file_name))

            dataset = ImageDataset(directory, image_size, aug_prob=0)
            dataloader = DataLoader(dataset, batch_size=batch_size)

            image_batch = next(iter(dataloader)).cuda(rank)
            images_augment = aug_wrapper(images=image_batch, prob=1, types=types, detach=True)

            save_result = file_name + f'_augs{ext}'
            torchvision.utils.save_image(images_augment, save_result, nrow=nrow)

            print('Save result to:', save_result)

        else:
            print('File not found. File', file)
