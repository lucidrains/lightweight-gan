import os
import json
import multiprocessing
from random import random
import math
from math import log2, floor
from functools import partial
from contextlib import contextmanager, ExitStack
from pathlib import Path
from shutil import rmtree

import torch
from torch.optim import Adam
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import grad as torch_grad
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from PIL import Image
import torchvision
from torchvision import transforms

from lightweight_gan.diff_augment import DiffAugment
from lightweight_gan.version import __version__

from tqdm import tqdm
from einops import rearrange
from pytorch_fid import fid_score

from hamburger_pytorch import Hamburger

# asserts

assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'

# constants

EPS = 1e-8
NUM_CORES = multiprocessing.cpu_count()
EXTS = ['jpg', 'jpeg', 'png']
CALC_FID_NUM_IMAGES = 12800

# helpers

def exists(val):
    return val is not None

@contextmanager
def null_context():
    yield

def combine_contexts(contexts):
    @contextmanager
    def multi_contexts():
        with ExitStack() as stack:
            yield [stack.enter_context(ctx()) for ctx in contexts]
    return multi_contexts

def default(val, d):
    return val if exists(val) else d

def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool

def cycle(iterable):
    while True:
        for i in iterable:
            yield i

def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException

def gradient_accumulate_contexts(gradient_accumulate_every, is_ddp, ddps):
    if is_ddp:
        num_no_syncs = gradient_accumulate_every - 1
        head = [combine_contexts(map(lambda ddp: ddp.no_sync, ddps))] * num_no_syncs
        tail = [null_context]
        contexts =  head + tail
    else:
        contexts = [null_context] * gradient_accumulate_every

    for context in contexts:
        with context():
            yield

def gradient_penalty(images, outputs, weight = 10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs=outputs, inputs=images,
                           grad_outputs=list(map(lambda t: torch.ones(t.size(), device = images.device), outputs)),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.reshape(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)

def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res

# activation classes

class Activation(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x)

class MishFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x_tanh_sp = torch.nn.functional.softplus(x).tanh()
        if x.requires_grad:
            ctx.save_for_backward(x_tanh_sp + x * x.sigmoid() * (1 - x_tanh_sp.square()))
        y = x * x_tanh_sp
        return y

    @staticmethod
    def backward(ctx, grad_output):
        if len(ctx.saved_tensors) == 0:
            return None
        grad, = ctx.saved_tensors
        return grad_output * grad
    
    
class SwishFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        with torch.no_grad():
            sigmoid_i = i.sigmoid()
            result = i * sigmoid_i
            if i.requires_grad:
                grad = sigmoid_i + result * (1 - sigmoid_i)
                ctx.save_for_backward(grad)
        result.requires_grad_(i.requires_grad)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        grad, = ctx.saved_tensors
        return grad_output * grad

def select_activation(name):
    fn = {
        'swish': SwishFn.apply,
        'mish': MishFn.apply,
        'elu': F.elu,
        'relu': F.relu,
        'gelu': F.gelu,
        'leaky_relu': partial(F.leaky_relu, negative_slope = 0.1)
    }[name.lower()]
    return lambda: Activation(fn)

# evonorm

def group_std(x, groups = 32, eps = 1e-5):  
    shape = x.shape 
    x = rearrange(x, 'b (g c) h w -> b g c h w', g = groups)    
    var = torch.var(x, dim = (2, 3, 4), keepdim = True).expand_as(x)    
    return torch.reshape(torch.sqrt(var + eps), shape)  

class EvoNorm2d(nn.Module): 
    def __init__(   
        self,   
        input,  
        groups = 32,
        eps = 1e-5, 
    ):  
        super().__init__()  
        self.eps = eps  
        self.groups = groups
        self.gamma = nn.Parameter(torch.ones(1, input, 1, 1))   
        self.beta = nn.Parameter(torch.zeros(1, input, 1, 1))   
        self.v = nn.Parameter(torch.ones(1, input, 1, 1))   

    def forward(self, x):   
        num = x * torch.sigmoid(self.v * x) 
        return num / group_std(x, groups = self.groups, eps = self.eps) * self.gamma + self.beta

# helper classes

class NanException(Exception):
    pass

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    def update_average(self, old, new):
        if not exists(old):
            return new
        return old * self.beta + (1 - self.beta) * new

class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else = lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob
    def forward(self, x):
        fn = self.fn if random() < self.prob else self.fn_else
        return fn(x)

# dataset

def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

class expand_greyscale(object):
    def __init__(self, transparent):
        self.transparent = transparent

    def __call__(self, tensor):
        channels = tensor.shape[0]
        num_target_channels = 4 if self.transparent else 3

        if channels == num_target_channels:
            return tensor

        alpha = None
        if channels == 1:
            color = tensor.expand(3, -1, -1)
        elif channels == 2:
            color = tensor[:1].expand(3, -1, -1)
            alpha = tensor[1:]
        else:
            raise Exception(f'image with invalid number of channels given {channels}')

        if not exists(alpha) and self.transparent:
            alpha = torch.ones(1, *tensor.shape[1:], device=tensor.device)

        return color if not self.transparent else torch.cat((color, alpha))

def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return torchvision.transforms.functional.resize(image, min_size)
    return image

class ImageDataset(Dataset):
    def __init__(self, folder, image_size, transparent = False, aug_prob = 0.):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in EXTS for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        assert len(self.paths) > 0, f'No images were found in {folder} for training'

        convert_image_fn = partial(convert_image_to, 'RGBA' if transparent else 'RGB')
        num_channels = 3 if not transparent else 4

        self.transform = transforms.Compose([
            transforms.Lambda(convert_image_fn),
            transforms.Lambda(partial(resize_to_minimum_size, image_size)),
            transforms.Resize(image_size),
            RandomApply(aug_prob, transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)), transforms.CenterCrop(image_size)),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale(transparent))
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# augmentations

def random_hflip(tensor, prob):
    if prob > random():
        return tensor
    return torch.flip(tensor, dims=(3,))

class AugWrapper(nn.Module):
    def __init__(self, D, image_size):
        super().__init__()
        self.D = D

    def forward(self, images, prob = 0., types = [], detach = False, **kwargs):
        if random() < prob:
            images = random_hflip(images, prob=0.5)
            images = DiffAugment(images, types=types)

        if detach:
            images = images.detach()

        return self.D(images, **kwargs)

# modifiable global variables

activation_fn = select_activation('leaky_relu')
norm_class = nn.BatchNorm2d
conv2d = nn.Conv2d

# classes

class SLE(nn.Module):
    def __init__(
        self,
        *,
        chan_in,
        chan_out
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(chan_in, chan_in, 4),
            activation_fn(),
            nn.Conv2d(chan_in, chan_out, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class Generator(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        latent_dim = 256,
        fmap_max = 512,
        fmap_inverse_coef = 12,
        transparent = False,
        hamburger_res_layers = []
    ):
        super().__init__()
        resolution = log2(image_size)
        assert resolution.is_integer(), 'image size must be a power of 2'
        init_channel = 4 if transparent else 3
        fmap_max = default(fmap_max, latent_dim)

        self.initial_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, latent_dim * 2, 4),
            norm_class(latent_dim * 2),
            nn.GLU(dim = 1)
        )

        num_layers = int(resolution) - 2
        features = list(map(lambda n: (n,  2 ** (fmap_inverse_coef - n)), range(2, num_layers + 2)))
        features = list(map(lambda n: (n[0], min(n[1], fmap_max)), features))
        features = list(map(lambda n: 3 if n[0] >= 8 else n[1], features))
        features = [latent_dim, *features]

        in_out_features = list(zip(features[:-1], features[1:]))

        self.res_layers = range(2, num_layers + 2)
        self.layers = nn.ModuleList([])
        self.res_to_feature_map = dict(zip(self.res_layers, in_out_features))

        self.sle_map = ((3, 7), (4, 8), (5, 9), (6, 10))
        self.sle_map = list(filter(lambda t: t[0] in self.res_layers and t[1] in self.res_layers, self.sle_map))
        self.sle_map = dict(self.sle_map)

        for (resolution, (chan_in, chan_out)) in zip(self.res_layers, in_out_features):
            image_width = 2 ** resolution

            hamburger = None
            if image_width in hamburger_res_layers:
                hamburger = Hamburger(
                    dim = chan_in,
                    n = image_width ** 2
                )

            sle = None
            if resolution in self.sle_map:
                residual_layer = self.sle_map[resolution]
                sle_chan_out = self.res_to_feature_map[residual_layer][-1]

                sle = SLE(
                    chan_in = chan_out,
                    chan_out = sle_chan_out
                )

            layer = nn.ModuleList([
                nn.Sequential(
                    nn.Upsample(scale_factor = 2),
                    conv2d(chan_in, chan_out * 2, 3, padding = 1),
                    norm_class(chan_out * 2),
                    nn.GLU(dim = 1)
                ),
                sle,
                hamburger
            ])
            self.layers.append(layer)

        self.out_conv = nn.Conv2d(features[-1], init_channel, 3, padding = 1)

    def forward(self, x):
        x = rearrange(x, 'b c -> b c () ()')
        x = self.initial_conv(x)

        residuals = dict()

        for (res, (up, sle, hamburger)) in zip(self.res_layers, self.layers):
            if exists(hamburger):
                x = hamburger(x) + x

            x = up(x)

            if exists(sle):
                out_res = self.sle_map[res]
                residual = sle(x)
                residuals[out_res] = residual

            if res in residuals:
                x = x * residuals[res]

        return self.out_conv(x)

class SimpleDecoder(nn.Module):
    def __init__(
        self,
        *,
        chan_in,
        chan_out = 3,
        num_upsamples = 4,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        final_chan = chan_out
        chans = chan_in

        for ind in range(num_upsamples):
            last_layer = ind == (num_upsamples - 1)
            chan_out = chans if not last_layer else final_chan * 2
            layer = nn.Sequential(
                nn.Upsample(scale_factor = 2),
                conv2d(chans, chan_out, 3, padding = 1),
                nn.GLU(dim = 1)
            )
            self.layers.append(layer)
            chans //= 2

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Discriminator(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        fmap_max = 512,
        fmap_inverse_coef = 12,
        transparent = False,
        hamburger_res_layers = []
    ):
        super().__init__()
        resolution = log2(image_size)
        assert resolution.is_integer(), 'image size must be a power of 2'
        init_channel = 4 if transparent else 3

        num_non_residual_layers = max(0, int(resolution) - 8)
        num_residual_layers = 8 - 3

        features = list(map(lambda n: (n,  2 ** (fmap_inverse_coef - n)), range(8, 2, -1)))
        features = list(map(lambda n: (n[0], min(n[1], fmap_max)), features))

        if num_non_residual_layers == 0:
            res, _ = features[0]
            features[0] = (res, init_channel)

        chan_in_out = zip(features[:-1], features[1:])

        self.non_residual_layers = nn.ModuleList([])
        for ind in range(num_non_residual_layers):
            first_layer = ind == 0
            last_layer = ind == (num_non_residual_layers - 1)
            chan_out = features[0][-1] if last_layer else init_channel

            self.non_residual_layers.append(nn.Sequential(
                conv2d(init_channel, chan_out, 4, stride = 2, padding = 1),
                activation_fn()
            ))

        self.residual_layers = nn.ModuleList([])
        for (res, ((_, chan_in), (_, chan_out))) in zip(range(8, 2, -1), chan_in_out):
            image_width = 2 ** resolution

            hamburger = None
            if image_width in hamburger_res_layers:
                hamburger = Hamburger(
                    dim = chan_in,
                    n = image_width ** 2
                )

            self.residual_layers.append(nn.ModuleList([
                nn.Sequential(
                    conv2d(chan_in, chan_out, 4, stride = 2, padding = 1),
                    activation_fn(),
                    conv2d(chan_out, chan_out, 3, padding = 1),
                    activation_fn()
                ),
                nn.Sequential(
                    nn.AvgPool2d(2),
                    conv2d(chan_in, chan_out, 1),
                    activation_fn()
                ),
                hamburger
            ]))

        last_chan = features[-1][-1]
        self.to_logits = nn.Sequential(
            conv2d(last_chan, last_chan, 1),
            activation_fn(),
            nn.Conv2d(last_chan, 1, 4)
        )

        self.decoder1 = SimpleDecoder(chan_in = last_chan, chan_out = init_channel)
        self.decoder2 = SimpleDecoder(chan_in = features[-2][-1], chan_out = init_channel)

    def forward(self, x, calc_aux_loss = False):
        orig_img = x

        for layer in self.non_residual_layers:
            x = layer(x)

        layer_outputs = []

        for (layer, residual_layer, hamburger) in self.residual_layers:
            if exists(hamburger):
                x = hamburger(x) + x

            x = layer(x) + residual_layer(x)
            layer_outputs.append(x)

        out = self.to_logits(x).flatten(1)

        if not calc_aux_loss:
            return out, None

        # self-supervised auto-encoding loss

        layer_8x8 = layer_outputs[-1]
        layer_16x16 = layer_outputs[-2]

        recon_img_8x8 = self.decoder1(layer_8x8)

        aux_loss1 = F.mse_loss(
            recon_img_8x8,
            F.interpolate(orig_img, size = recon_img_8x8.shape[2:])
        )

        select_random_quadrant = lambda rand_quadrant, img: rearrange(img, 'b c (m h) (n w) -> (m n) b c h w', m = 2, n = 2)[rand_quadrant]
        crop_image_fn = partial(select_random_quadrant, floor(random() * 4))
        img_part, layer_16x16_part = map(crop_image_fn, (orig_img, layer_16x16))

        recon_img_16x16 = self.decoder2(layer_16x16_part)

        aux_loss2 = F.mse_loss(
            recon_img_16x16,
            F.interpolate(img_part, size = recon_img_16x16.shape[2:])
        )

        aux_loss = aux_loss1 + aux_loss2

        return out, aux_loss

class LightweightGAN(nn.Module):
    def __init__(
        self,
        *,
        latent_dim,
        image_size,
        fmap_max = 512,
        fmap_inverse_coef = 12,
        transparent = False,
        hamburger_res_layers = [],
        ttur_mult = 1.5,
        lr = 2e-4,
        rank = 0,
        ddp = False
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        G_kwargs = dict(
            image_size = image_size,
            latent_dim = latent_dim,
            fmap_max = fmap_max,
            fmap_inverse_coef = fmap_inverse_coef,
            transparent = transparent,
            hamburger_res_layers = hamburger_res_layers
        )

        self.G = Generator(**G_kwargs)

        self.D = Discriminator(
            image_size = image_size,
            fmap_max = fmap_max,
            fmap_inverse_coef = fmap_inverse_coef,
            transparent = transparent,
            hamburger_res_layers = hamburger_res_layers
        )

        self.ema_updater = EMA(0.995)
        self.GE = Generator(**G_kwargs)
        set_requires_grad(self.GE, False)


        self.G_opt = Adam(self.G.parameters(), lr = lr, betas=(0.5, 0.9))
        self.D_opt = Adam(self.D.parameters(), lr = lr * ttur_mult, betas=(0.5, 0.9))

        self.apply(self._init_weights)
        self.reset_parameter_averaging()

        self.cuda(rank)
        self.D_aug = AugWrapper(self.D, image_size)

    def _init_weights(self, m):
        if type(m) in {nn.Conv2d, nn.Linear}:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

            for current_buffer, ma_buffer in zip(current_model.buffers(), ma_model.buffers()):
                new_buffer_value = self.ema_updater.update_average(ma_buffer, current_buffer)
                ma_buffer.copy_(new_buffer_value)

        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        raise NotImplemented

# trainer

class Trainer():
    def __init__(
        self,
        name = 'default',
        results_dir = 'results',
        models_dir = 'models',
        base_dir = './',
        latent_dim = 256,
        image_size = 128,
        fmap_max = 512,
        transparent = False,
        batch_size = 4,
        mixed_prob = 0.9,
        gradient_accumulate_every = 1,
        hamburger_res_layers = [],
        use_evonorm = False,
        lr = 2e-4,
        lr_mlp = 1.,
        ttur_mult = 2,
        save_every = 1000,
        evaluate_every = 1000,
        trunc_psi = 0.6,
        aug_prob = 0.,
        aug_types = ['translation', 'cutout'],
        dataset_aug_prob = 0.,
        calculate_fid_every = None,
        is_ddp = False,
        rank = 0,
        world_size = 1,
        log = False,
        activation = 'leaky_relu',
        *args,
        **kwargs
    ):
        self.GAN_params = [args, kwargs]
        self.GAN = None

        self.name = name

        base_dir = Path(base_dir)
        self.base_dir = base_dir
        self.results_dir = base_dir / results_dir
        self.models_dir = base_dir / models_dir
        self.config_path = self.models_dir / name / '.config.json'

        assert log2(image_size).is_integer(), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.fmap_max = fmap_max
        self.transparent = transparent

        self.aug_prob = aug_prob
        self.aug_types = aug_types

        self.lr = lr
        self.ttur_mult = ttur_mult
        self.batch_size = batch_size
        self.mixed_prob = mixed_prob

        self.evaluate_every = evaluate_every
        self.save_every = save_every
        self.steps = 0

        self.gradient_accumulate_every = gradient_accumulate_every

        self.activation = activation
        self.use_evonorm = use_evonorm
        self.hamburger_res_layers = hamburger_res_layers

        self.d_loss = 0
        self.g_loss = 0
        self.last_gp_loss = None
        self.last_recon_loss = None
        self.last_fid = None

        self.init_folders()

        self.loader = None
        self.dataset_aug_prob = dataset_aug_prob

        self.calculate_fid_every = calculate_fid_every

        self.is_ddp = is_ddp
        self.is_main = rank == 0
        self.rank = rank
        self.world_size = world_size

        self.syncbatchnorm = is_ddp

    @property
    def image_extension(self):
        return 'jpg' if not self.transparent else 'png'

    @property
    def checkpoint_num(self):
        return floor(self.steps // self.save_every)
        
    def init_GAN(self):
        args, kwargs = self.GAN_params

        # set some global variables before instantiating GAN

        global activation_fn
        global norm_class

        activation_fn = select_activation(self.activation)
        norm_class = nn.BatchNorm2d if not self.use_evonorm else lambda chans: EvoNorm2d(chans, groups = 32) if chans >= 64 else nn.Identity()
        norm_class = nn.SyncBatchNorm if not self.use_evonorm and self.syncbatchnorm else norm_class

        # handle bugs when
        # switching from multi-gpu back to single gpu

        if self.syncbatchnorm and not self.is_ddp:
            import torch.distributed as dist
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group('nccl', rank=0, world_size=1)

        # instantiate GAN

        self.GAN = LightweightGAN(
            lr = self.lr,
            latent_dim = self.latent_dim,
            hamburger_res_layers = self.hamburger_res_layers,
            image_size = self.image_size,
            ttur_mult = self.ttur_mult,
            fmap_max = self.fmap_max,
            transparent = self.transparent,
            rank = self.rank,
            *args,
            **kwargs
        )

        if self.is_ddp:
            ddp_kwargs = {'device_ids': [self.rank], 'output_device': self.rank, 'find_unused_parameters': True}

            self.G_ddp = DDP(self.GAN.G, **ddp_kwargs)
            self.D_ddp = DDP(self.GAN.D, **ddp_kwargs)
            self.D_aug_ddp = DDP(self.GAN.D_aug, **ddp_kwargs)

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
        self.image_size = config['image_size']
        self.transparent = config['transparent']
        self.use_evonorm = config['use_evonorm']
        self.hamburger_res_layers = config['hamburger_res_layers']
        self.use_evonorm = config['use_evonorm']
        self.activation = config['activation']
        self.syncbatchnorm = config['syncbatchnorm']
        self.fmap_max = config.pop('fmap_max', 512)
        del self.GAN
        self.init_GAN()

    def config(self):
        return {
            'image_size': self.image_size,
            'transparent': self.transparent,
            'use_evonorm': self.use_evonorm,
            'syncbatchnorm': self.syncbatchnorm,
            'hamburger_res_layers': self.hamburger_res_layers,
            'activation': self.activation
        }

    def set_data_src(self, folder):
        self.dataset = ImageDataset(folder, self.image_size, transparent = self.transparent, aug_prob = self.dataset_aug_prob)
        sampler = DistributedSampler(self.dataset, rank=self.rank, num_replicas=self.world_size, shuffle=True) if self.is_ddp else None
        dataloader = DataLoader(self.dataset, num_workers = NUM_CORES, batch_size = math.ceil(self.batch_size / self.world_size), sampler = sampler, shuffle = not self.is_ddp, drop_last = True, pin_memory = True)
        self.loader = cycle(dataloader)

    def train(self):
        assert exists(self.loader), 'You must first initialize the data source with `.set_data_src(<folder of images>)`'

        if not exists(self.GAN):
            self.init_GAN()

        self.GAN.train()
        total_disc_loss = torch.tensor(0.).cuda(self.rank)
        total_gen_loss = torch.tensor(0.).cuda(self.rank)

        batch_size = math.ceil(self.batch_size / self.world_size)

        image_size = self.GAN.image_size
        latent_dim = self.GAN.latent_dim

        aug_prob   = self.aug_prob
        aug_types  = self.aug_types
        aug_kwargs = {'prob': aug_prob, 'types': aug_types}

        G = self.GAN.G if not self.is_ddp else self.G_ddp
        D = self.GAN.D if not self.is_ddp else self.D_ddp
        D_aug = self.GAN.D_aug if not self.is_ddp else self.D_aug_ddp

        apply_gradient_penalty = self.steps % 4 == 0

        # train discriminator

        self.GAN.D_opt.zero_grad()

        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[D_aug, G]):
            latents = torch.randn(batch_size, latent_dim).cuda(self.rank)

            generated_images = G(latents)
            fake_output, fake_aux_loss = D_aug(generated_images.clone().detach(), calc_aux_loss = True, detach = True, **aug_kwargs)

            image_batch = next(self.loader).cuda(self.rank)
            image_batch.requires_grad_()
            real_output, real_aux_loss = D_aug(image_batch,  calc_aux_loss = True, **aug_kwargs)

            real_output_loss = real_output
            fake_output_loss = fake_output

            divergence = (F.relu(1 + real_output_loss) + F.relu(1 - fake_output_loss)).mean()
            disc_loss = divergence

            aux_loss = real_aux_loss + fake_aux_loss
            self.last_recon_loss = aux_loss.clone().detach().item()
            disc_loss = disc_loss + aux_loss

            if apply_gradient_penalty:
                gp = gradient_penalty(image_batch, (real_output,))
                self.last_gp_loss = gp.clone().detach().item()
                disc_loss = disc_loss + gp

            disc_loss = disc_loss / self.gradient_accumulate_every
            disc_loss.register_hook(raise_if_nan)
            disc_loss.backward()

            total_disc_loss += divergence.detach().item() / self.gradient_accumulate_every

        self.d_loss = float(total_disc_loss)

        self.GAN.D_opt.step()

        # train generator

        self.GAN.G_opt.zero_grad()

        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[G, D_aug]):
            latents = torch.randn(batch_size, latent_dim).cuda(self.rank)
            generated_images = G(latents)
            fake_output, _ = D_aug(generated_images, **aug_kwargs)
            fake_output_loss = fake_output

            loss = fake_output_loss.mean()
            gen_loss = loss

            gen_loss = gen_loss / self.gradient_accumulate_every
            gen_loss.register_hook(raise_if_nan)
            gen_loss.backward()

            total_gen_loss += loss.detach().item() / self.gradient_accumulate_every

        self.g_loss = float(total_gen_loss)

        self.GAN.G_opt.step()

        # calculate moving averages

        if self.is_main and self.steps % 10 == 0 and self.steps > 20000:
            self.GAN.EMA()

        if self.is_main and self.steps <= 25000 and self.steps % 1000 == 2:
            self.GAN.reset_parameter_averaging()

        # save from NaN errors

        if any(torch.isnan(l) for l in (total_gen_loss, total_disc_loss)):
            print(f'NaN detected for generator or discriminator. Loading from checkpoint #{self.checkpoint_num}')
            self.load(self.checkpoint_num)
            raise NanException

        # periodically save results

        if self.is_main:
            if self.steps % self.save_every == 0:
                self.save(self.checkpoint_num)

            if self.steps % self.evaluate_every == 0 or (self.steps % 100 == 0 and self.steps < 2500):
                self.evaluate(floor(self.steps / self.evaluate_every))

            if exists(self.calculate_fid_every) and self.steps % self.calculate_fid_every == 0 and self.steps != 0:
                num_batches = math.ceil(CALC_FID_NUM_IMAGES / self.batch_size)
                fid = self.calculate_fid(num_batches)
                self.last_fid = fid

                with open(str(self.results_dir / self.name / f'fid_scores.txt'), 'a') as f:
                    f.write(f'{self.steps},{fid}\n')

        self.steps += 1

    @torch.no_grad()
    def evaluate(self, num = 0, num_image_tiles = 8, trunc = 1.0):
        self.GAN.eval()

        ext = self.image_extension
        num_rows = num_image_tiles
    
        latent_dim = self.GAN.latent_dim
        image_size = self.GAN.image_size

        # latents and noise

        latents = torch.randn(num_rows ** 2, latent_dim).cuda(self.rank)

        # regular

        generated_images = self.generate_truncated(self.GAN.G, latents)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}.{ext}'), nrow=num_rows)
        
        # moving averages

        generated_images = self.generate_truncated(self.GAN.GE, latents)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-ema.{ext}'), nrow=num_rows)

    @torch.no_grad()
    def calculate_fid(self, num_batches):
        torch.cuda.empty_cache()

        real_path = str(self.results_dir / self.name / 'fid_real') + '/'
        fake_path = str(self.results_dir / self.name / 'fid_fake') + '/'

        # remove any existing files used for fid calculation and recreate directories
        rmtree(real_path, ignore_errors=True)
        rmtree(fake_path, ignore_errors=True)
        os.makedirs(real_path)
        os.makedirs(fake_path)

        for batch_num in tqdm(range(num_batches), desc='calculating FID - saving reals'):
            real_batch = next(self.loader)
            for k in range(real_batch.size(0)):
                torchvision.utils.save_image(real_batch[k, :, :, :], real_path + '{}.png'.format(k + batch_num * self.batch_size))

        # generate a bunch of fake images in results / name / fid_fake
        self.GAN.eval()
        ext = self.image_extension

        latent_dim = self.GAN.latent_dim
        image_size = self.GAN.image_size

        for batch_num in tqdm(range(num_batches), desc='calculating FID - saving generated'):
            # latents and noise
            latents = torch.randn(self.batch_size, latent_dim).cuda(self.rank)

            # moving averages
            generated_images = self.generate_truncated(self.GAN.GE, latents)

            for j in range(generated_images.size(0)):
                torchvision.utils.save_image(generated_images[j, :, :, :], str(Path(fake_path) / f'{str(j + batch_num * self.batch_size)}-ema.{ext}'))

        return fid_score.calculate_fid_given_paths([real_path, fake_path], 256, True, 2048)

    @torch.no_grad()
    def generate_truncated(self, G, style, trunc_psi = 0.75, num_image_tiles = 8):
        generated_images = evaluate_in_chunks(self.batch_size, G, style)
        return generated_images.clamp_(0., 1.)

    @torch.no_grad()
    def generate_interpolation(self, num = 0, num_image_tiles = 8, trunc = 1.0, num_steps = 100, save_frames = False):
        self.GAN.eval()
        ext = self.image_extension
        num_rows = num_image_tiles

        latent_dim = self.GAN.latent_dim
        image_size = self.GAN.image_size

        # latents and noise

        latents_low = torch.randn(num_rows ** 2, latent_dim).cuda(self.rank)
        latents_high = torch.randn(num_rows ** 2, latent_dim).cuda(self.rank)

        ratios = torch.linspace(0., 8., num_steps)

        frames = []
        for ratio in tqdm(ratios):
            interp_latents = slerp(ratio, latents_low, latents_high)
            generated_images = self.generate_truncated(self.GAN.GE, interp_latents)
            images_grid = torchvision.utils.make_grid(generated_images, nrow = num_rows)
            pil_image = transforms.ToPILImage()(images_grid.cpu())
            
            if self.transparent:
                background = Image.new('RGBA', pil_image.size, (255, 255, 255))
                pil_image = Image.alpha_composite(background, pil_image)
                
            frames.append(pil_image)

        frames[0].save(str(self.results_dir / self.name / f'{str(num)}.gif'), save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)

        if save_frames:
            folder_path = (self.results_dir / self.name / f'{str(num)}')
            folder_path.mkdir(parents=True, exist_ok=True)
            for ind, frame in enumerate(frames):
                frame.save(str(folder_path / f'{str(ind)}.{ext}'))

    def print_log(self):
        data = [
            ('G', self.g_loss),
            ('D', self.d_loss),
            ('GP', self.last_gp_loss),
            ('SS', self.last_recon_loss),
            ('FID', self.last_fid)
        ]

        data = [d for d in data if exists(d[1])]
        log = ' | '.join(map(lambda n: f'{n[0]}: {n[1]:.2f}', data))
        print(log)

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    def clear(self):
        rmtree(str(self.models_dir / self.name), True)
        rmtree(str(self.results_dir / self.name), True)
        rmtree(str(self.config_path), True)
        self.init_folders()

    def save(self, num):
        save_data = {
            'GAN': self.GAN.state_dict(),
            'version': __version__
        }

        torch.save(save_data, self.model_name(num))
        self.write_config()

    def load(self, num = -1):
        self.load_config()

        name = num
        if num == -1:
            file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pt')]
            saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f'continuing from previous epoch - {name}')

        self.steps = name * self.save_every

        load_data = torch.load(self.model_name(name))

        if 'version' in load_data and self.is_main:
            print(f"loading from version {load_data['version']}")

        try:
            self.GAN.load_state_dict(load_data['GAN'])
        except Exception as e:
            print('unable to load save model. please try downgrading the package to the version specified by the saved model')
            raise e
