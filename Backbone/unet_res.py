import torch
import math
import torch.nn as nn
from utility.SinusoidalPosEmb import SinusoidalPositionalEmbedding

def Normalize(dim):
    return nn.GroupNorm(num_groups=32, num_channels=dim, eps=1e-6, affine=True)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class ResnetBlock(nn.Module):
    def __init__(self,
                 *,
                 in_channel,
                 out_channel=None,
                 conv_shortcut=False,
                 dropout=0.,
                 term_channel=512):
        super().__init__()
        self.in_channel = in_channel
        out_channel = out_channel if out_channel is not None else in_channel
        self.out_channel = out_channel
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channel)
        self.conv1 = nn.Conv2d(in_channel, 
                               out_channel, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1)
        self.norm2 = Normalize(out_channel)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channel, 
                               out_channel, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1)
        
        if in_channel != out_channel:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channel, 
                                               out_channel, 
                                               kernel_size=3, 
                                               stride=1, 
                                               padding=0)
            else:
                self.nin_shortcut = nn.Conv2d(in_channel, 
                                              out_channel, 
                                              kernel_size=1, 
                                              stride=1, 
                                              padding=0)
    def forward(self, x):
        in_x = x
        x = self.norm1(x)
        x = torch.relu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        if self.in_channel != self.out_channel:
            if self.use_conv_shortcut:
                in_x = self.conv_shortcut(in_x)
            else:
                in_x = self.nin_shortcut(in_x)
        return x + in_x
    
class AttnBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.norm = Normalize(in_channel)
        self.in_channel = in_channel
        self.q = nn.Conv2d(in_channel, in_channel,
                            kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channel, in_channel,
                            kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channel, in_channel,
                            kernel_size=1, stride=1, padding=0)
        self.project_out = nn.Conv2d(in_channel, in_channel,
                                        kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        h = x
        h = self.norm(h)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        #compute att
        b,c,h,w =q.shape
        q = q.reshape(b,c,h*w).permute(0,2,1) # B, N, C
        k = k.reshape(b,c,h*w) # B, C, N
        weight = torch.bmm(q,k) # B, N, N
        #scale
        weight = weight * int(c) ** (-0.5)
        weight = torch.nn.functional.softmax(weight, dim=2)

        #attention to value
        v = v.reshape(b,c,h*w)
        weight = weight.permute(0,2,1) # B, N, N || 1st is k and 2nd is q
        h = torch.bmm(v, weight) # B, C, H(of q) ->H_[b,c,j] = sum_i(v[b,c,i] * weight[b,i,j])
        h = h.reshape(b,c,h,w)

        h = self.project_out(h)

        return h + x
    
class UnetResNetBlock(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, with_time_emb=True,
                 in_channels, resolution):
        super().__init__()
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.with_time_emb = with_time_emb

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


        temb = torch.relu(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):elu(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = torch.relu(h)
        h = self.conv_out(h)
        return h