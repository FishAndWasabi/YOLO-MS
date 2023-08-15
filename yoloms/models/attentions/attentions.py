import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from mmyolo.registry import MODELS
from ..utils import autopad
from mmcv.cnn import ConvModule

from typing import Sequence, Union
from mmdet.utils import ConfigType

@MODELS.register_module()
class SE(nn.Module):
    def __init__(self, dim=1, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) 
        y = self.fc(y).view(b, c, 1, 1) 
        return x * y.expand_as(x)


@MODELS.register_module()
class MDTA(nn.Module):
    def __init__(self, dim, num_heads=4,bias=False):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        b,c,h,w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


@MODELS.register_module()
class CA(nn.Module):
    def __init__(self, dim, reduction=16):
        super(CA, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        mip = max(8, dim // reduction)
        self.conv1 = nn.Conv2d(dim, mip, kernel_size=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(mip, dim, kernel_size=1)
    def forward(self, x):
        identity = x
        n,c,h,w = x.size()
        x = self.pool(x)
        x = self.conv1(x)
        x = self.act(x) 
        x = self.conv2(x)
        out = identity * x.sigmoid()
        return out