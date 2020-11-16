import torch
import torch.nn as nn
from einops import rearrange

def group_std(x, groups = 32, eps = 1e-5):
    shape = x.shape
    x = rearrange(x, 'b (g c) h w -> b g c h w', g = groups)
    var = torch.var(x, dim = (2, 3, 4), keepdim = True).expand_as(x)
    return torch.reshape(torch.sqrt(var + eps), shape)

class EvoNorm2d(nn.Module):
    def __init__(
        self,
        input,
        eps = 1e-5,
        groups = 32,
        min_channels = 32
    ):
        super().__init__()
        self.groups = groups if input > min_channels else 1
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(1, input, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, input, 1, 1))
        self.v = nn.Parameter(torch.ones(1, input, 1, 1))

    def forward(self, x):
        num = x * torch.sigmoid(self.v * x)
        return num / group_std(x, groups = self.groups, eps = self.eps) * self.gamma + self.beta
