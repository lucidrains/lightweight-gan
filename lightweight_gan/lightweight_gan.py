import torch
from math import log2
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

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
        num_layers = log2(image_size)
        assert num_layers.is_integer(), 'image size must be a power of 2'
        fmap_max = default(fmap_max, latent_dim)

        self.initial_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, latent_dim * 2, 4),
            nn.BatchNorm2d(latent_dim * 2),
            nn.GLU(dim = 1)
        )

        num_layers = int(num_layers) - 2
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
            sle = SLE(
                chan_in = chan_out,
                chan_out = self.res_to_feature_map[self.sle_map[resolution]][-1]
            ) if resolution in self.sle_map else None

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

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class LightweightGAN(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x
