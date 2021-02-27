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
from torch.cuda.amp import autocast, GradScaler
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
from kornia import filter2D

from diff_augment import DiffAugment
from version import __version__
from bn import CategoricalConditionalBatchNorm2d

from tqdm import tqdm
from einops import rearrange, reduce

from adabelief_pytorch import AdaBelief
from gsa_pytorch import GSA

# asserts

assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'

# constants

NUM_CORES = multiprocessing.cpu_count()
EXTS = ['jpg', 'jpeg', 'png']

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


def is_power_of_two(val):
    return log2(val).is_integer()


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
        head = [combine_contexts(
            map(lambda ddp: ddp.no_sync, ddps))] * num_no_syncs
        tail = [null_context]
        contexts = head + tail
    else:
        contexts = [null_context] * gradient_accumulate_every

    for context in contexts:
        with context():
            yield


def hinge_loss(real, fake):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()


def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(
        zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    chunked_classes = [x[1] for x in chunked_outputs]  # TODO: return it?
    chunked_outputs = [x[0] for x in chunked_outputs]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)


def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * \
        low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res


def safe_div(n, d):
    try:
        res = n / d
    except ZeroDivisionError:
        prefix = '' if int(n >= 0) else '-'
        res = float(f'{prefix}inf')
    return res

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
    def __init__(self, prob, fn, fn_else=lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob

    def forward(self, x):
        fn = self.fn if random() < self.prob else self.fn_else
        return fn(x)


class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x):
        return self.g * self.fn(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class SumBranches(nn.Module):
    def __init__(self, branches):
        super().__init__()
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        return sum(map(lambda fn: fn(x), self.branches))


class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2D(x, f, normalized=True)

# dataset


def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


class identity(object):
    def __call__(self, tensor):
        return tensor


# augmentations

def random_hflip(tensor, prob):
    if prob > random():
        return tensor
    return torch.flip(tensor, dims=(3,))


class AugWrapper(nn.Module):
    def __init__(self, D, image_size):
        super().__init__()
        self.D = D

    def forward(self, images, y=None, prob=0., types=[], detach=False, **kwargs):
        context = torch.no_grad if detach else null_context

        with context():
            if random() < prob:
                images = random_hflip(images, prob=0.5)
                images = DiffAugment(images, types=types)

        return self.D(images, y, **kwargs)

# modifiable global variables


# TODO: make sure this is placeholder
norm_class = nn.BatchNorm2d


def upsample(scale_factor=2):
    return nn.Upsample(scale_factor=scale_factor)

# squeeze excitation classes

# global context network
# https://arxiv.org/abs/2012.13375
# similar to squeeze-excite, but with a simplified attention pooling and a subsequent layer norm


class GlobalContext(nn.Module):
    def __init__(
        self,
        *,
        chan_in,
        chan_out
    ):
        super().__init__()
        self.to_k = nn.Conv2d(chan_in, 1, 1)
        chan_intermediate = max(3, chan_out // 2)

        self.net = nn.Sequential(
            nn.Conv2d(chan_in, chan_intermediate, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(chan_intermediate, chan_out, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        context = self.to_k(x)
        context = context.flatten(2).softmax(dim=-1)
        out = einsum('b i n, b c n -> b c i', context, x.flatten(2))
        out = out.unsqueeze(-1)
        return self.net(out)

# frequency channel attention
# https://arxiv.org/abs/2012.11879


def get_1d_dct(i, freq, L):
    result = math.cos(math.pi * freq * (i + 0.5) / L) / math.sqrt(L)
    return result * (1 if freq == 0 else math.sqrt(2))


def get_dct_weights(width, channel, fidx_u, fidx_v):
    dct_weights = torch.zeros(1, channel, width, width)
    c_part = channel // len(fidx_u)

    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)):
        for x in range(width):
            for y in range(width):
                coor_value = get_1d_dct(x, u_x, width) * \
                    get_1d_dct(y, v_y, width)
                dct_weights[:, i * c_part: (i + 1) * c_part, x, y] = coor_value

    return dct_weights


class FCANet(nn.Module):
    def __init__(
        self,
        *,
        chan_in,
        chan_out,
        reduction=4,
        width
    ):
        super().__init__()

        # in paper, it seems 16 frequencies was ideal
        freq_w, freq_h = ([0] * 8), list(range(8))
        dct_weights = get_dct_weights(
            width, chan_in, [*freq_w, *freq_h], [*freq_h, *freq_w])
        self.register_buffer('dct_weights', dct_weights)

        chan_intermediate = max(3, chan_out // reduction)

        self.net = nn.Sequential(
            nn.Conv2d(chan_in, chan_intermediate, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(chan_intermediate, chan_out, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = reduce(x * self.dct_weights,
                   'b c (h h1) (w w1) -> b c h1 w1', 'sum', h1=1, w1=1)
        return self.net(x)


# generative adversarial network
EMBEDDING_DIM = 16


class InitConv(nn.Module):
    def __init__(self, latent_dim, num_classes=0, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.c1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, latent_dim * 2, 4),
            norm_class(latent_dim * 2),
            nn.GLU(dim=1)
        )
        self.num_classes = num_classes
        if num_classes == 0:
            return
        raise NotImplementedError

        self.embed = nn.Embedding(num_classes, embedding_dim)
        self.c2 = nn.Sequential(
            nn.Conv2d(latent_dim+embedding_dim, latent_dim*2, 1),
            norm_class(latent_dim * 2),
            nn.GLU(dim=1)
        )

    def forward(self, x, y=None):
        left = self.c1(x)
        if y is None:
            if self.num_classes == 0:
                return left
            else:
                y = torch.randint(self.num_classes, x.shape[:1], device="cuda")
        assert self.num_classes > 0
        right = self.embed(y)[:, :, None, None].repeat(1, 1, 4, 4,)
        return self.c2(torch.cat((left, right), 1))


class GenSeq(nn.Module):
    def __init__(self, chan_in, chan_out, num_classes=0):
        super().__init__()
        if num_classes > 0:
            self.norm = CategoricalConditionalBatchNorm2d(
                num_classes, chan_out*2)
        else:
            self.norm = norm_class(chan_out * 2)

        self.prenorm = nn.Sequential(upsample(), Blur(), nn.Conv2d(
            chan_in, chan_out * 2, 3, padding=1))
        self.postnorm = nn.GLU(dim=1)

    def forward(self, x, y=None):
        x = self.prenorm(x)
        x = self.norm(x) if y is None else self.norm(x, y)
        return self.postnorm(x)
    
    
class Catter(nn.Module):
    def __init__(self, feat_dim, embedding_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.integrate = nn.Sequential(
            nn.Conv2d(feat_dim+embedding_dim, feat_dim*2, 1),
            norm_class(feat_dim * 2),
            nn.GLU(dim=1)
        )
        
    def forward(self, x, y=None):
        im_width = x.shape[-1]
        assert im_width == x.shape[-2], "is image a square?"
        embedded = self.embedding(y)[:, :, None, None].repeat(1, 1, im_width, im_width,)
        return self.integrate(torch.cat((x, embedded), 1))



class Generator(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        latent_dim=256,
        fmap_max=512,
        fmap_inverse_coef=12,
        num_chans=3,
        attn_res_layers=[],
        freq_chan_attn=False,
        num_classes=0,
        cat_res_layers=[],
    ):
        super().__init__()
        assert num_classes > 0 or cat_res_layers == []
        resolution = log2(image_size)
        assert is_power_of_two(image_size), 'image size must be a power of 2'

        init_channel = num_chans

        fmap_max = default(fmap_max, latent_dim)

        self.init_conv = InitConv(latent_dim, 0)  # HERE
        self.num_classes = num_classes

        num_layers = int(resolution) - 2
        features = list(
            map(lambda n: (n,  2 ** (fmap_inverse_coef - n)), range(2, num_layers + 2)))
        features = list(map(lambda n: (n[0], min(n[1], fmap_max)), features))
        features = list(map(lambda n: 3 if n[0] >= 8 else n[1], features))  # TODO: should it be num chans?
        features = [latent_dim, *features]

        in_out_features = list(zip(features[:-1], features[1:]))

        self.res_layers = range(2, num_layers + 2)
        self.layers = nn.ModuleList([])
        self.res_to_feature_map = dict(zip(self.res_layers, in_out_features))

        self.sle_map = ((3, 7), (4, 8), (5, 9), (6, 10))
        self.sle_map = list(
            filter(lambda t: t[0] <= resolution and t[1] <= resolution, self.sle_map))
        self.sle_map = dict(self.sle_map)

        self.num_layers_spatial_res = 1

        for (res, (chan_in, chan_out)) in zip(self.res_layers, in_out_features):
            image_width = 2 ** res

            cat = Catter(chan_in, EMBEDDING_DIM, num_classes) if image_width in cat_res_layers else None
            attn = None
            if image_width in attn_res_layers:
                attn = Rezero(GSA(dim=chan_in, norm_queries=True))
                

            sle = None
            if res in self.sle_map:
                residual_layer = self.sle_map[res]
                sle_chan_out = self.res_to_feature_map[residual_layer - 1][-1]

                if freq_chan_attn:
                    sle = FCANet(
                        chan_in=chan_out,
                        chan_out=sle_chan_out,
                        width=2 ** (res + 1)
                    )
                else:
                    sle = GlobalContext(
                        chan_in=chan_out,
                        chan_out=sle_chan_out
                    )

            layer = nn.ModuleList([
                cat,
                GenSeq(chan_in, chan_out, 0),
                sle,
                attn
            ])
            self.layers.append(layer)

        self.out_conv = nn.Conv2d(features[-1], init_channel, 3, padding=1)

    def forward(self, x, y=None):
        x = rearrange(x, 'b c -> b c () ()')
        x = self.init_conv(x, y)
        x = F.normalize(x, dim=1)

        residuals = dict()
        if self.num_classes > 0 and y is None:
            y = torch.randint(self.num_classes, x.shape[:1], device="cuda")
        for (res, (cat, up, sle, attn)) in zip(self.res_layers, self.layers):
            if exists(cat):
                x = cat(x,y)
            if exists(attn):
                x = attn(x) + x

            x = up(x)  #, y)

            if exists(sle):
                out_res = self.sle_map[res]
                residual = sle(x)
                residuals[out_res] = residual

            next_res = res + 1
            if next_res in residuals:
                x = x * residuals[next_res]

        return self.out_conv(x), y


class SimpleDecoder(nn.Module):
    def __init__(
        self,
        *,
        chan_in,
        chan_out=3,
        num_upsamples=4,
        end_glu=True,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        final_chan = chan_out
        chans = chan_in

        for ind in range(num_upsamples):
            last_layer = ind == (num_upsamples - 1)
            chan_out = chans if (not last_layer or end_glu) else final_chan * 2
            layer = nn.Sequential(
                upsample(),
                nn.Conv2d(chans, chan_out, 3, padding=1),
                nn.GLU(dim=1)
            )
            self.layers.append(layer)
            chans //= 2

        if end_glu:
            self.layers.append(nn.Conv2d(chans, final_chan, 3, padding=1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        fmap_max=512,
        fmap_inverse_coef=12,
        num_chans=3,
        disc_output_size=5,
        attn_res_layers=[],
        num_classes=0,
        bn4decoder=False,
        projection_loss_scale=1
    ):
        super().__init__()
        resolution = log2(image_size)
        assert is_power_of_two(image_size), 'image size must be a power of 2'
        assert disc_output_size in {
            1, 5}, 'discriminator output dimensions can only be 5x5 or 1x1'

        resolution = int(resolution)

        init_channel = num_chans

        num_non_residual_layers = max(0, int(resolution) - 8)
        num_residual_layers = 8 - 3

        non_residual_resolutions = range(min(8, resolution), 2, -1)
        features = list(
            map(lambda n: (n,  2 ** (fmap_inverse_coef - n)), non_residual_resolutions))
        features = list(map(lambda n: (n[0], min(n[1], fmap_max)), features))

        if num_non_residual_layers == 0:
            res, _ = features[0]
            features[0] = (res, init_channel)

        chan_in_out = list(zip(features[:-1], features[1:]))

        self.non_residual_layers = nn.ModuleList([])
        for ind in range(num_non_residual_layers):
            first_layer = ind == 0
            last_layer = ind == (num_non_residual_layers - 1)
            chan_out = features[0][-1] if last_layer else init_channel

            self.non_residual_layers.append(nn.Sequential(
                Blur(),
                nn.Conv2d(init_channel, chan_out, 4, stride=2, padding=1),
                nn.LeakyReLU(0.1)
            ))

        self.residual_layers = nn.ModuleList([])

        for (res, ((_, chan_in), (_, chan_out))) in zip(non_residual_resolutions, chan_in_out):
            image_width = 2 ** res  # res vs resolution

            attn = None
            if image_width in attn_res_layers:
                attn = Rezero(
                    GSA(dim=chan_in, batch_norm=False, norm_queries=True))

            self.residual_layers.append(nn.ModuleList([
                SumBranches([
                    nn.Sequential(
                        Blur(),
                        nn.Conv2d(chan_in, chan_out, 4, stride=2, padding=1),
                        nn.LeakyReLU(0.1),
                        nn.Conv2d(chan_out, chan_out, 3, padding=1),
                        nn.LeakyReLU(0.1)
                    ),
                    nn.Sequential(
                        Blur(),
                        nn.AvgPool2d(2),
                        nn.Conv2d(chan_in, chan_out, 1),
                        nn.LeakyReLU(0.1),
                    )
                ]),
                attn
            ]))

        last_chan = features[-1][-1]
        if disc_output_size == 5:
            raise NotImplementedError  # though on projection
            self.to_pre_logits = nn.Sequential(
                nn.Conv2d(last_chan, last_chan, 1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(last_chan, 1, 4)
            )
        elif disc_output_size == 1:
            self.to_pre_logits = nn.Sequential(
                Blur(),
                nn.Conv2d(last_chan, last_chan, 3, stride=2, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(last_chan, last_chan, 4),
                nn.LeakyReLU(0.1),
            )
            self.out = nn.Linear(last_chan, 1)

        self.to_shape_disc_out = nn.Sequential(
            nn.Conv2d(init_channel, 64, 3, padding=1),
            Residual(Rezero(GSA(dim=64, norm_queries=True, batch_norm=False))),
            SumBranches([
                nn.Sequential(
                    Blur(),
                    nn.Conv2d(64, 32, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(32, 32, 3, padding=1),
                    nn.LeakyReLU(0.1)
                ),
                nn.Sequential(
                    Blur(),
                    nn.AvgPool2d(2),
                    nn.Conv2d(64, 32, 1),
                    nn.LeakyReLU(0.1),
                )
            ]),
            Residual(Rezero(GSA(dim=32, norm_queries=True, batch_norm=False))),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(32, 1, 4)
        )

        self.decoder1 = SimpleDecoder(
            chan_in=last_chan, chan_out=init_channel, end_glu=bn4decoder)
        self.decoder2 = SimpleDecoder(
            chan_in=features[-2][-1], chan_out=init_channel, end_glu=bn4decoder) if resolution >= 9 else None

        if num_classes > 0:
            self.l_y = nn.utils.spectral_norm(
                nn.Embedding(num_classes, last_chan))
        self._initialize()

        self.bn4decoder = nn.BatchNorm2d(
            num_chans) if bn4decoder else nn.Identity()  # GLU enforces not affine
        self.projection_loss_scale = projection_loss_scale

    def _initialize(self):
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            nn.init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None, calc_aux_loss=False):
        orig_img = x

        for layer in self.non_residual_layers:
            x = layer(x)

        layer_outputs = []

        for (net, attn) in self.residual_layers:
            if exists(attn):
                x = attn(x) + x

            x = net(x)
            layer_outputs.append(x)

        x = self.to_pre_logits(x).flatten(1)
        out = self.out(x)

        # HERE V
        if y is not None:
            out += torch.sum(self.l_y(y) * x, dim=1, keepdim=True) * self.projection_loss_scale

        img_32x32 = F.interpolate(orig_img, size=(32, 32))
        out_32x32 = self.to_shape_disc_out(img_32x32)

        if not calc_aux_loss:
            return out, out_32x32, None

        # self-supervised auto-encoding loss

        layer_8x8 = layer_outputs[-1]
        layer_16x16 = layer_outputs[-2]

        recon_img_8x8 = self.decoder1(layer_8x8)

        aux_loss = F.mse_loss(
            recon_img_8x8,
            F.interpolate(self.bn4decoder(orig_img),
                          size=recon_img_8x8.shape[2:])
        )

        if exists(self.decoder2):
            def select_random_quadrant(rand_quadrant, img): return rearrange(
                img, 'b c (m h) (n w) -> (m n) b c h w', m=2, n=2)[rand_quadrant]
            crop_image_fn = partial(
                select_random_quadrant, floor(random() * 4))
            img_part, layer_16x16_part = map(
                crop_image_fn, (orig_img, layer_16x16))

            recon_img_16x16 = self.decoder2(layer_16x16_part)

            aux_loss_16x16 = F.mse_loss(
                recon_img_16x16,
                F.interpolate(self.bn4decoder(img_part),
                              size=recon_img_16x16.shape[2:])
            )

            aux_loss = aux_loss + aux_loss_16x16

        # TODO: output vs aux loss? I think output, generator is bakcproped on it
        return out, out_32x32, aux_loss


class LightweightGAN(nn.Module):
    def __init__(
        self,
        *,
        latent_dim,
        image_size,
        optimizer="adam",
        fmap_max=512,
        fmap_inverse_coef=12,
        num_chans=3,
        disc_output_size=5,
        attn_res_layers=[],
        freq_chan_attn=False,
        ttur_mult=1.,
        lr=2e-4,
        rank=0,
        ddp=False,
        num_classes=0,
        projection_loss_scale=1
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        G_kwargs = dict(
            image_size=image_size,
            latent_dim=latent_dim,
            fmap_max=fmap_max,
            fmap_inverse_coef=fmap_inverse_coef,
            num_chans=num_chans,
            attn_res_layers=attn_res_layers,
            freq_chan_attn=freq_chan_attn,
            num_classes=num_classes,
        )

        self.G = Generator(**G_kwargs)

        self.D = Discriminator(
            image_size=image_size,
            fmap_max=fmap_max,
            fmap_inverse_coef=fmap_inverse_coef,
            num_chans=num_chans,
            attn_res_layers=attn_res_layers,
            disc_output_size=disc_output_size,
            num_classes=num_classes,
            projection_loss_scale=projection_loss_scale,
        )

        self.ema_updater = EMA(0.995)
        self.GE = Generator(**G_kwargs)
        set_requires_grad(self.GE, False)

        if optimizer == "adam":
            self.G_opt = Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.9))
            self.D_opt = Adam(self.D.parameters(), lr=lr *
                              ttur_mult, betas=(0.5, 0.9))
        elif optimizer == "adabelief":
            self.G_opt = AdaBelief(self.G.parameters(),
                                   lr=lr, betas=(0.5, 0.9))
            self.D_opt = AdaBelief(self.D.parameters(),
                                   lr=lr * ttur_mult, betas=(0.5, 0.9))
        else:
            assert False, "No valid optimizer is given"

        self.apply(self._init_weights)
        self.reset_parameter_averaging()

        self.cuda(rank)
        self.D_aug = AugWrapper(self.D, image_size)

    def _init_weights(self, m):
        if type(m) in {nn.Conv2d, nn.Linear}:
            nn.init.kaiming_normal_(
                m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(
                    old_weight, up_weight)

            for current_buffer, ma_buffer in zip(current_model.buffers(), ma_model.buffers()):
                new_buffer_value = self.ema_updater.update_average(
                    ma_buffer, current_buffer)
                ma_buffer.copy_(new_buffer_value)

        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, latents, detach_gen=True, **aug_kwargs):
        context = torch.no_grad if detach_gen else null_context
        with context():
            generated_images, y = self.G(latents)

        return self.D_aug(
            generated_images, y, detach=detach_gen, **aug_kwargs)


# trainer


class Trainer():
    def __init__(
        self,
        name='default',
        results_dir='results',
        models_dir='models',
        base_dir='./',
        optimizer='adam',
        num_workers=None,
        latent_dim=256,
        image_size=128,
        num_image_tiles=8,
        fmap_max=512,
        num_chans=3,
        batch_size=4,
        gp_weight=10,
        gradient_accumulate_every=1,
        attn_res_layers=[],
        freq_chan_attn=False,
        disc_output_size=5,
        antialias=False,
        lr=2e-4,
        lr_mlp=1.,
        ttur_mult=1.,
        save_every=1000,
        evaluate_every=1000,
        aug_prob=None,
        aug_types=['translation', 'cutout'],
        dataset_aug_prob=0.,
        rank=0,
        world_size=1,
        multi_gpus=False,
        num_classes=0,
        aux_loss_multi=0.04,
        projection_loss_scale=1,
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

        assert is_power_of_two(
            image_size), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        assert all(map(is_power_of_two, attn_res_layers)
                   ), 'resolution layers of attention must all be powers of 2 (16, 32, 64, 128, 256, 512)'

        self.image_size = image_size
        self.num_image_tiles = num_image_tiles

        self.latent_dim = latent_dim
        self.fmap_max = fmap_max
        self.num_chans = num_chans

        self.aug_prob = aug_prob
        self.aug_types = aug_types

        self.lr = lr
        self.optimizer = optimizer
        self.num_workers = num_workers
        self.ttur_mult = ttur_mult
        self.batch_size = batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.gp_weight = gp_weight

        self.evaluate_every = evaluate_every
        self.save_every = save_every
        self.steps = 0

        self.generator_top_k_gamma = 0.99
        self.generator_top_k_frac = 0.5

        self.attn_res_layers = attn_res_layers
        self.freq_chan_attn = freq_chan_attn

        self.disc_output_size = disc_output_size
        self.antialias = antialias

        self.d_loss = 0
        self.g_loss = 0
        self.last_gp_loss = None
        self.last_recon_loss = None

        self.init_folders()

        self.loader = None
        self.dataset_aug_prob = dataset_aug_prob

        self.is_main = rank == 0
        self.rank = rank
        self.world_size = world_size
        self.multi_gpus = multi_gpus
        self.num_classes = num_classes
        self.aux_loss_multi = aux_loss_multi
        self.projection_loss_scale = projection_loss_scale

    @property
    def image_extension(self):
        return 'jpg'

    @property
    def checkpoint_num(self):
        return floor(self.steps // self.save_every)

    def init_GAN(self):
        args, kwargs = self.GAN_params

        # set some global variables before instantiating GAN

        global norm_class
        global Blur

        # TODO: KOKO
        norm_class = nn.BatchNorm2d
        Blur = nn.Identity if not self.antialias else Blur

        # handle bugs when
        # switching from multi-gpu back to single gpu

        # instantiate GAN

        self.GAN = LightweightGAN(
            optimizer=self.optimizer,
            lr=self.lr,
            latent_dim=self.latent_dim,
            attn_res_layers=self.attn_res_layers,
            freq_chan_attn=self.freq_chan_attn,
            image_size=self.image_size,
            ttur_mult=self.ttur_mult,
            fmap_max=self.fmap_max,
            disc_output_size=self.disc_output_size,
            rank=self.rank,
            num_classes=self.num_classes,
            projection_loss_scale=self.projection_loss_scale,
            *args,
            **kwargs
        )
        self.parallel()

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = self.config() if not self.config_path.exists(
        ) else json.loads(self.config_path.read_text())
        self.image_size = config['image_size']
        self.disc_output_size = config['disc_output_size']
        self.num_chans = config.pop('num_chans', False)
        self.attn_res_layers = config.pop('attn_res_layers', [])
        self.freq_chan_attn = config.pop('freq_chan_attn', False)
        self.optimizer = config.pop('optimizer', 'adam')
        self.fmap_max = config.pop('fmap_max', 512)
        del self.GAN
        self.init_GAN()

    def config(self):
        return {
            'image_size': self.image_size,
            'num_chans': self.num_chans,
            'disc_output_size': self.disc_output_size,
            'optimizer': self.optimizer,
            'attn_res_layers': self.attn_res_layers,
            'freq_chan_attn': self.freq_chan_attn
        }

    def set_dataset(self, dataset, num_workers):
        self.dataset = dataset
        dataloader = DataLoader(self.dataset, num_workers=num_workers,
                                batch_size=self.batch_size, drop_last=True, pin_memory=True)
        self.loader = cycle(dataloader)

    def parallel(self):
        self.parallel_D_aug = nn.DataParallel(
            self.GAN.D_aug) if self.multi_gpus else self.GAN.D_aug
        self.parallel_GD = nn.DataParallel(
            self.GAN) if self.multi_gpus else self.GAN

    def train(self):
        assert exists(
            self.loader), 'You must first initialize the data source with `.set_data_src(<folder of images>)`'
        device = torch.device(f'cuda:{self.rank}')

        if not exists(self.GAN):
            self.init_GAN()

        self.GAN.train()
        total_disc_loss = torch.zeros([], device=device)
        total_gen_loss = torch.zeros([], device=device)

        batch_size = math.ceil(self.batch_size / self.world_size)

        image_size = self.GAN.image_size
        latent_dim = self.GAN.latent_dim

        aug_prob = default(self.aug_prob, 0)
        aug_types = self.aug_types
        aug_kwargs = {'prob': aug_prob, 'types': aug_types}

        G = self.GAN.G
        D = self.GAN.D
        D_aug = self.parallel_D_aug
        Y = self.parallel_GD

        apply_gradient_penalty = self.steps % 4 == 0

        # amp related contexts and functions

        # train discriminator
        self.GAN.D_opt.zero_grad()
        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, False, ddps=[D_aug, G]):
            latents = torch.randn(batch_size, latent_dim).cuda(self.rank)
            image_batch = next(self.loader)
            if self.num_classes > 0:
                assert type(image_batch) in {
                    tuple, list}, "Conditional GAN got no labels provided"
                image_batch, y = image_batch
                y = y.cuda(self.rank)
            else:
                y = None
            image_batch = image_batch.cuda(self.rank)
            image_batch.requires_grad_()

            fake_output, fake_output_32x32, _ = Y(latents, True, **aug_kwargs)
            fake_output = fake_output.mean(0)
            fake_output_32x32 = fake_output_32x32.mean(0)

            real_output, real_output_32x32, real_aux_loss = D_aug(
                image_batch, y, calc_aux_loss=True, **aug_kwargs)  # TODO: pass y from data here
            real_output = real_output.mean(0)
            real_output_32x32 = real_output_32x32.mean(0)
            real_aux_loss = real_aux_loss.mean(0)

            real_output_loss = real_output
            fake_output_loss = fake_output  # TODO: is this shape good?

            divergence = hinge_loss(real_output_loss, fake_output_loss)
            divergence_32x32 = hinge_loss(real_output_32x32, fake_output_32x32)
            disc_loss = divergence + divergence_32x32

            aux_loss = real_aux_loss * self.aux_loss_multi
            disc_loss = disc_loss + aux_loss

            if apply_gradient_penalty:
                outputs = [real_output, real_output_32x32]

                scaled_gradients = torch_grad(outputs=outputs, inputs=image_batch,
                                              grad_outputs=list(map(lambda t: torch.ones(
                                                  t.size(), device=image_batch.device), outputs)),
                                              create_graph=True, retain_graph=True, only_inputs=True)[0]

                inv_scale = 1.

                if inv_scale != float('inf'):
                    gradients = scaled_gradients * inv_scale

                    gradients = gradients.reshape(batch_size, -1)
                    gp = self.gp_weight * \
                        ((gradients.norm(2, dim=1) - 1) ** 2).mean()

                    if not torch.isnan(gp):
                        disc_loss = disc_loss + gp
                        self.last_gp_loss = gp.clone().detach().item()

            disc_loss = disc_loss / self.gradient_accumulate_every

            disc_loss.register_hook(raise_if_nan)
            disc_loss.backward()
            total_disc_loss += divergence

        self.last_recon_loss = aux_loss.item()
        self.d_loss = float(total_disc_loss.item() /
                            self.gradient_accumulate_every)
        self.GAN.D_opt.step()

        # train generator

        self.GAN.G_opt.zero_grad()
        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, False, ddps=[D_aug, G]):
            latents = torch.randn(batch_size, latent_dim).cuda(self.rank)

            fake_output, fake_output_32x32, _ = Y(latents, False, **aug_kwargs)

            fake_output_loss = fake_output.mean(
                dim=1) + fake_output_32x32.mean(dim=1)

            epochs = (self.steps * batch_size *
                      self.gradient_accumulate_every) / len(self.dataset)
            k_frac = max(self.generator_top_k_gamma **
                         epochs, self.generator_top_k_frac)
            k = math.ceil(batch_size * k_frac)

            if k != batch_size:
                fake_output_loss, _ = fake_output_loss.topk(k=k, largest=False)

            loss = fake_output_loss.mean()
            gen_loss = loss

            gen_loss = gen_loss / self.gradient_accumulate_every

            gen_loss.register_hook(raise_if_nan)
            gen_loss.backward()
            total_gen_loss += loss

        self.g_loss = float(total_gen_loss.item() /
                            self.gradient_accumulate_every)
        self.GAN.G_opt.step()

        # calculate moving averages

        if self.is_main and self.steps % 10 == 0 and self.steps > 20000:
            self.GAN.EMA()

        if self.is_main and self.steps <= 25000 and self.steps % 1000 == 2:
            self.GAN.reset_parameter_averaging()

        # save from NaN errors

        if any(torch.isnan(l) for l in (total_gen_loss, total_disc_loss)):
            print(
                f'NaN detected for generator or discriminator. Loading from checkpoint #{self.checkpoint_num}')
            self.load(self.checkpoint_num)
            raise NanException

        del total_disc_loss
        del total_gen_loss

        # periodically save results

        if self.is_main:
            if self.steps % self.save_every == 0:
                self.save(self.checkpoint_num)

            if self.steps % self.evaluate_every == 0 or (self.steps % 100 == 0 and self.steps < 20000):
                self.evaluate(floor(self.steps / self.evaluate_every),
                              num_image_tiles=self.num_image_tiles)

        self.steps += 1

    @torch.no_grad()
    def evaluate(self, num=0, num_image_tiles=4):
        self.GAN.eval()

        ext = self.image_extension
        num_rows = num_image_tiles

        latent_dim = self.GAN.latent_dim
        image_size = self.GAN.image_size

        # latents and noise

        latents = torch.randn((num_rows ** 2, latent_dim)).cuda(self.rank)

        # regular

        generated_images = self.generate_(self.GAN.G, latents)
        torchvision.utils.save_image(generated_images, str(
            self.results_dir / self.name / f'{str(num)}.{ext}'), nrow=num_rows, padding=4, pad_value=1)

        # moving averages

        generated_images = self.generate_(self.GAN.GE, latents)
        torchvision.utils.save_image(generated_images, str(
            self.results_dir / self.name / f'{str(num)}-ema.{ext}'), nrow=num_rows, padding=4, pad_value=1)

    @torch.no_grad()
    def generate(self, num=0, num_image_tiles=4, checkpoint=None, types=['default', 'ema']):
        self.GAN.eval()

        latent_dim = self.GAN.latent_dim
        dir_name = self.name + str('-generated-') + str(checkpoint)
        dir_full = Path().absolute() / self.results_dir / dir_name
        ext = self.image_extension

        if not dir_full.exists():
            os.mkdir(dir_full)

        # regular
        if 'default' in types:
            for i in tqdm(range(num_image_tiles), desc='Saving generated default images'):
                latents = torch.randn((1, latent_dim)).cuda(self.rank)
                generated_image = self.generate_(self.GAN.G, latents)
                path = str(self.results_dir / dir_name /
                           f'{str(num)}-{str(i)}.{ext}')
                torchvision.utils.save_image(generated_image[0], path, nrow=1)

        # moving averages
        if 'ema' in types:
            for i in tqdm(range(num_image_tiles), desc='Saving generated EMA images'):
                latents = torch.randn((1, latent_dim)).cuda(self.rank)
                generated_image = self.generate_(self.GAN.GE, latents)
                path = str(self.results_dir / dir_name /
                           f'{str(num)}-{str(i)}-ema.{ext}')
                torchvision.utils.save_image(generated_image[0], path, nrow=1)

        return dir_full

    @torch.no_grad()
    def show_progress(self, num_images=4, types=['default', 'ema']):
        checkpoints = self.get_checkpoints()
        assert exists(
            checkpoints), 'cannot find any checkpoints to create a training progress video for'

        dir_name = self.name + str('-progress')
        dir_full = Path().absolute() / self.results_dir / dir_name
        ext = self.image_extension
        latents = None

        if not dir_full.exists():
            os.mkdir(dir_full)

        for checkpoint in tqdm(checkpoints, desc='Generating progress images'):
            self.load(checkpoint, print_version=False)
            self.GAN.eval()

            if checkpoint == 0:
                latents = torch.randn(
                    (num_images, self.GAN.latent_dim)).cuda(self.rank)

            # regular
            if 'default' in types:
                generated_image = self.generate_(self.GAN.G, latents)
                path = str(self.results_dir / dir_name /
                           f'{str(checkpoint)}.{ext}')
                torchvision.utils.save_image(
                    generated_image, path, nrow=num_images)

            # moving averages
            if 'ema' in types:
                generated_image = self.generate_(self.GAN.GE, latents)
                path = str(self.results_dir / dir_name /
                           f'{str(checkpoint)}-ema.{ext}')
                torchvision.utils.save_image(
                    generated_image, path, nrow=num_images)

    @torch.no_grad()
    def generate_(self, G, style, num_image_tiles=8):
        generated_images = evaluate_in_chunks(self.batch_size, G, style)
        return generated_images.clamp_(0., 1.)

    @torch.no_grad()
    def generate_interpolation(self, num=0, num_image_tiles=8, num_steps=100, save_frames=False):
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
            generated_images = self.generate_(self.GAN.GE, interp_latents)
            images_grid = torchvision.utils.make_grid(
                generated_images, nrow=num_rows)
            pil_image = transforms.ToPILImage()(images_grid.cpu())

            frames.append(pil_image)

        frames[0].save(str(self.results_dir / self.name /
                           f'{str(num)}.gif'), save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)

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
            'version': __version__,
        }

        torch.save(save_data, self.model_name(num))
        self.write_config()

    def load(self, num=-1, print_version=True):
        self.load_config()

        name = num
        if num == -1:
            checkpoints = self.get_checkpoints()

            if not exists(checkpoints):
                return

            name = checkpoints[-1]
            print(f'continuing from previous epoch - {name}')

        self.steps = name * self.save_every

        load_data = torch.load(self.model_name(name))

        if print_version and 'version' in load_data and self.is_main:
            print(f"loading from version {load_data['version']}")

        try:
            self.GAN.load_state_dict(load_data['GAN'])
        except Exception as e:
            print('unable to load save model. please try downgrading the package to the version specified by the saved model')
            raise e

    def get_checkpoints(self):
        file_paths = [p for p in Path(
            self.models_dir / self.name).glob('model_*.pt')]
        saved_nums = sorted(
            map(lambda x: int(x.stem.split('_')[1]), file_paths))

        if len(saved_nums) == 0:
            return None

        return saved_nums
