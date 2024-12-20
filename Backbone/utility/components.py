import torch.nn as nn
from inspect import isfunction

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, stride=2, padding=1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, stride=2, padding=1)

