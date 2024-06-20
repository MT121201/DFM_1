import sys
import math
import torch
import torch.nn as nn
from einops import rearrange

from utility.SinusoidalPosEmb import SinusoidalPositionalEmbedding
from utility.Residual import Residual
from utility.Normalization import PreNorm, LayerNorm
from utility.components import *
from Attention.LinearAttention import LinearAttention

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
    

class UnetConvNextBlock(nn.Module):
    def __init__(self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        with_time_emb = True,
        output_mean_scale = False,
        residual = False,
        ):
        super().__init__()
        self.channels = channels
        self.residual = residual
        print("Is Time embed used ? ", with_time_emb)
        self.output_mean_scale = output_mean_scale
        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPositionalEmbedding(dim),
                nn.Linear(dim, dim *4),
                nn.GELU(),
                nn.Linear(dim * 4, time_dim))
            
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out) 

        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx >= num_resolutions -1 
            self.downs.append(nn.ModuleList([
                ConvNextBlock(dim_in, dim_out, time_emb = time_dim, norm = idx !=0),
                ConvNextBlock(dim_out, dim_out, time_emb = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
                ]))
            
        mid_dim = dims[-1]
        self.mid_block1 =  ConvNextBlock(mid_dim, mid_dim, time_emb = time_dim)
        self.mid_attn =  Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 =  ConvNextBlock(mid_dim, mid_dim, time_emb = time_dim)

        for idx, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = idx >= num_resolutions -1
            self.ups.append(nn.ModuleList([
                ConvNextBlock(dim_out *2, dim_in, time_emb = time_dim),
                ConvNextBlock(dim_in, dim_in, time_emb = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            ConvNextBlock(dim, dim, time_emb = time_dim),
            nn.Conv2d(dim, out_dim, 1))
        
    def forward(self, x, time=None):  
        orginal_x = x
        t = None
        if time is not None and exists(self.time_mlp):
            t = self.time_mlp(time)

        orginal_mean = torch.mean(x, dim = [1, 2, 3], keepdim = True)
        h=[]

        for convnext1, convnext2, attn, downsample in self.downs:
            x = convnext1(x, t)
            x = convnext2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x= self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for convnext1, convnext2, attn, upsample in self.ups:
            x = upsample(x)
            x = torch.cat((x, h.pop()), dim = 1)
            x = convnext1(x, t)
            x = convnext2(x, t)
            x = attn(x)
            x = upsample(x)

        if self.residual:
            return self.final_conv(x) + orginal_x
        
        out = self.final_conv(x)
        if self.output_mean_scale:
            out_mean = torch.mean(out, dim = [1, 2, 3], keepdim = True)
            out = out - orginal_mean + out_mean

        return out