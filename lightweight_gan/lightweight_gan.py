import torch
from torch.optim import Adam
import torch.nn.functional as F

import multiprocessing
from random import random
from math import log2, floor
from functools import partial
from torch import nn, einsum
from einops import rearrange

# asserts

assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'

# constants

NUM_CORES = multiprocessing.cpu_count()

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool

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
            nn.LeakyReLU(0.1),
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
        fmap_inverse_coef = 12
    ):
        super().__init__()
        resolution = log2(image_size)
        assert resolution.is_integer(), 'image size must be a power of 2'
        fmap_max = default(fmap_max, latent_dim)

        self.initial_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, latent_dim * 2, 4),
            nn.BatchNorm2d(latent_dim * 2),
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
                    nn.Conv2d(chan_in, chan_out * 2, 3, padding = 1),
                    nn.BatchNorm2d(chan_out * 2),
                    nn.GLU(dim = 1)
                ),
                sle
            ])
            self.layers.append(layer)

        self.out_conv = nn.Conv2d(features[-1], 3, 3, padding = 1)

    def forward(self, x):
        x = rearrange(x, 'b c -> b c () ()')
        x = self.initial_conv(x)

        residuals = dict()

        for (res, (up, sle)) in zip(self.res_layers, self.layers):
            x = up(x)
            if exists(sle):
                out_res = self.sle_map[res]
                residual = sle(x)
                residuals[out_res] = residual

            if res in residuals:
                x = x + residuals[res]

        x = self.out_conv(x)
        return x.tanh()

class SimpleDecoder(nn.Module):
    def __init__(
        self,
        *,
        chan_in,
        num_upsamples = 4
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        chans = chan_in
        for ind in range(num_upsamples):
            last_layer = ind == (num_upsamples - 1)
            chan_out = chans if not last_layer else 3 * 2
            layer = nn.Sequential(
                nn.Upsample(scale_factor = 2),
                nn.Conv2d(chans, chan_out, 3, padding = 1),
                nn.BatchNorm2d(chan_out),
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
        fmap_inverse_coef = 12
    ):
        super().__init__()
        resolution = log2(image_size)
        assert resolution.is_integer(), 'image size must be a power of 2'

        num_non_residual_layers = max(0, int(resolution) - 8)
        num_residual_layers = 8 - 3

        features = list(map(lambda n: (n,  2 ** (fmap_inverse_coef - n)), range(8, 2, -1)))
        features = list(map(lambda n: (n[0], min(n[1], fmap_max)), features))
        chan_in_out = zip(features[:-1], features[1:])

        self.non_residual_layers = nn.ModuleList([])
        for ind in range(num_non_residual_layers):
            first_layer = ind == 0
            last_layer = ind == (num_non_residual_layers - 1)
            chan_out = features[0][-1] if last_layer else 3

            self.non_residual_layers.append(nn.Sequential(
                nn.Conv2d(3, chan_out, 4, stride = 2, padding = 1),
                nn.BatchNorm2d(3) if not first_layer else nn.Identity(),
                nn.LeakyReLU(0.1)
            ))

        self.residual_layers = nn.ModuleList([])
        for (_, chan_in), (_, chan_out) in chan_in_out:
            self.residual_layers.append(nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(chan_in, chan_out, 4, stride = 2, padding = 1),
                    nn.BatchNorm2d(chan_out),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(chan_out, chan_out, 3, padding = 1),
                    nn.BatchNorm2d(chan_out),
                    nn.LeakyReLU(0.1)
                ),
                nn.Sequential(
                    nn.AvgPool2d(2),
                    nn.Conv2d(chan_in, chan_out, 1),
                    nn.BatchNorm2d(chan_out),
                    nn.LeakyReLU(0.1)
                ),
            ]))

        last_chan = features[-1][-1]
        self.to_logits = nn.Sequential(
            nn.Conv2d(last_chan, last_chan, 1),
            nn.BatchNorm2d(last_chan),
            nn.LeakyReLU(0.1),
            nn.Conv2d(last_chan, 1, 4)
        )

        self.decoder1 = SimpleDecoder(chan_in = last_chan)
        self.decoder2 = SimpleDecoder(chan_in = features[-2][-1])

    def forward(self, x):
        orig_img = x

        for layer in self.non_residual_layers:
            x = layer(x)

        layer_outputs = []

        for (layer, residual_layer) in self.residual_layers:
            x = layer(x) + residual_layer(x)
            layer_outputs.append(x)

        out = self.to_logits(x)

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

        return out.flatten(1), aux_loss

class LightweightGAN(nn.Module):
    def __init__(
        self,
        *,
        latent_dim,
        image_size,
        fmap_max = 512,
        fmap_inverse_coef = 12,
        lr = 1e-4
    ):
        super().__init__()
        G_kwargs = dict(image_size = image_size, latent_dim = latent_dim, fmap_max = fmap_max, fmap_inverse_coef = fmap_inverse_coef)
        self.G = Generator(**G_kwargs)
        self.GE = Generator(**G_kwargs)
        set_requires_grad(self.GE, False)

        self.D = Discriminator(image_size = image_size, fmap_max = fmap_max, fmap_inverse_coef = fmap_inverse_coef)

        self.G_opt = Adam(self.G.parameters(), lr = lr, betas=(0.5, 0.9))
        self.D_opt = Adam(self.D.parameters(), lr = lr * 2, betas=(0.5, 0.9))

        self.cuda()

    def forward(self, x):
        raise NotImplemented
