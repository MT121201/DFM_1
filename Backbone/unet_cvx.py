import sys
import math
import torch
import torch.nn as nn
from einops import rearrange

from utility.SinusoidalPosEmb import SinusoidalPositionalEmbedding
from utility.Residual import Residual
from utility.Normalization import PreNorm, LayerNorm
from utility.components import *

class ConvNextBlock(nn.Module):
    """ https://arxiv.org/abs/2201.03545 """
    def __init__(self, dim,
                dim_out,
                *,
                time_emb=None,
                mult=2,
                norm=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb, dim) if exists(time_emb) else None)
        # 7×7 depthwise 
        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.net = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim * mult, 3 , padding=1),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1)
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if exists(self.mlp):
            assert exists(time_emb), 'Time embedding must be provided'
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, 'b c -> b c 1 1')

        h = self.net(h)
        return h + self.res_conv(x)