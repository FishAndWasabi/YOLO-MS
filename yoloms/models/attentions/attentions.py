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
class PA(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads=4, 
                 qkv_bias=False, 
                 qk_scale=None, 
                 attn_drop=0., 
                 proj_drop=0., 
                 pool_ratios=[5,5,5],
                 use_conv=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.num_elements = np.array([t*t for t in pool_ratios]).sum()
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.pool_ratios = pool_ratios
        if isinstance(self.pool_ratios,int):
            self.pool_ratios = [self.pool_ratios]
        
        self.pools = nn.ModuleList()
        
        self.norm = nn.LayerNorm(dim)
        
        
        self.d_convs = [None for _ in pool_ratios]
        if use_conv:
            self.d_convs = nn.ModuleList([nn.Conv2d(dim, 
                                                    dim, 
                                                    kernel_size=3, 
                                                    stride=1, 
                                                    padding=1, 
                                                    groups=dim) for _ in pool_ratios])

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W              
        q = self.q(x.reshape(B, C, -1).permute(0,2,1)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        pools = []

        for (pool_ratio, l) in zip(self.pool_ratios, self.d_convs):
            pool = F.adaptive_avg_pool2d(x, (round(H/pool_ratio), round(W/pool_ratio)))
            if l is not None:
                pool = pool + l(pool)
            pools.append(pool.view(B, C, -1))
        
        pools = torch.cat(pools, dim=2)
        pools = self.norm(pools.permute(0,2,1))
        
        kv = self.kv(pools).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v)   
        x = x.transpose(1,2).contiguous().reshape(B, N, C)
        x = self.proj(x)        
        x = x.permute(0,2,1).reshape(B, C, H, W)
        return x

@MODELS.register_module()
class PAv2(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 pool_ratios: Union[int, Sequence[int]] =[1,2,3,6],
                 conv_cfg: ConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 expand_ratio=0.5,
                 num_heads=2, 
                 qkv_bias=False, 
                 qk_scale=None, 
                 attn_drop=0., 
                 proj_drop=0.):
        super().__init__()
        assert in_channels % num_heads == 0, f"dim {in_channels} should be divided by num_heads {num_heads}."
        self.mid_channels = int(in_channels * expand_ratio)
        self.out_channels = out_channels
        
        self.in_conv = ConvModule(in_channels, 
                                  self.mid_channels, 
                                  1, 
                                  conv_cfg=conv_cfg, 
                                  norm_cfg=norm_cfg,
                                  act_cfg=act_cfg)
        
        self.out_conv = ConvModule(self.mid_channels,
                                   self.out_channels, 
                                   1, 
                                   conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg,
                                   act_cfg=act_cfg)
        
        self.num_heads = num_heads
        self.num_elements = np.array([t*t for t in pool_ratios]).sum()
        head_dim = self.mid_channels // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Sequential(nn.Linear(self.mid_channels, self.mid_channels, bias=qkv_bias))
        self.kv = nn.Sequential(nn.Linear(self.mid_channels, self.mid_channels * 2, bias=qkv_bias))
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.mid_channels, self.mid_channels)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool_ratios = pool_ratios
        self.pools = nn.ModuleList()
        
        self.norm = nn.LayerNorm(self.mid_channels)
        

    def forward(self, x):
        B, _, H, W = x.shape
        N = H * W
        x = self.in_conv(x)
        
        q = self.q(x.reshape(B, self.mid_channels, -1).permute(0,2,1))
        q = q.reshape(B, N, self.num_heads, self.mid_channels // self.num_heads).permute(0, 2, 1, 3)
        
        pools = []
        
        for pool_ratio in self.pool_ratios:
            pool = F.adaptive_avg_pool2d(x, (round(H/pool_ratio), round(W/pool_ratio)))
            pools.append(pool.view(B, self.mid_channels, -1))
        
        pools = torch.cat(pools, dim=2)
        pools = self.norm(pools.permute(0,2,1))
        
        kv = self.kv(pools).reshape(B, -1, 2, self.num_heads, self.mid_channels // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v)   
        x = x.transpose(1,2).contiguous().reshape(B, N, self.mid_channels)
        x = self.proj(x)        
        x = x.permute(0,2,1).reshape(B, self.mid_channels, H, W)
        
        x = self.out_conv(x)
        return x


@MODELS.register_module()
class PAv3(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 pool_ratios: Union[int, Sequence[int]] =[5,5,5,5],
                 kernel_sizes: Union[int, Sequence[int]] =[5,5,5,5],
                 conv_cfg: ConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 expand_ratio=2):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = self.in_channels // len(kernel_sizes)
        self.out_channels = out_channels
        self.expand_ratio = expand_ratio
        self.in_conv = ConvModule(in_channels, 
                                  in_channels, 
                                  1, 
                                  conv_cfg=conv_cfg, 
                                  norm_cfg=norm_cfg,
                                  act_cfg=act_cfg)
        
        self.poolings = []
        self.convs = []
        for kernel_size, pool_ratio in zip(kernel_sizes,pool_ratios):
            padding=autopad(pool_ratio)
            self.poolings.append(nn.MaxPool2d(kernel_size=pool_ratio, stride=1, padding=padding, ceil_mode=True))
            groups = int(self.mid_channels*self.expand_ratio)
            self.mid_proj_in = ConvModule(self.mid_channels,
                                          groups,
                                          1,
                                          conv_cfg=conv_cfg,
                                          act_cfg=act_cfg, 
                                          norm_cfg=norm_cfg)
            self.mid_conv = ConvModule(groups,
                                       groups,
                                       kernel_size,
                                       padding=autopad(kernel_size),
                                       groups=groups,
                                       conv_cfg=conv_cfg,
                                       act_cfg=act_cfg,
                                       norm_cfg=norm_cfg)
            self.mid_proj_out = ConvModule(groups,
                                       self.mid_channels,
                                       1,
                                       conv_cfg=conv_cfg,
                                       act_cfg=act_cfg, 
                                       norm_cfg=norm_cfg)
            self.convs.append(nn.Sequential(self.mid_proj_in ,
                                                self.mid_conv, 
                                                self.mid_proj_out))
            
        
        self.poolings = nn.ModuleList(self.poolings)
        self.convs = nn.ModuleList(self.convs)
        self.out_conv = ConvModule(self.in_channels,
                                   self.out_channels, 
                                   1, 
                                   conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg,
                                   act_cfg=act_cfg)
        

    def forward(self, x):
        B, _, H, W = x.shape
        N = H * W
        x = self.in_conv(x)
        pools = []
        for i, (pooling, conv) in enumerate(zip(self.poolings, self.convs)):
            channel = x[:,i*self.mid_channels:(i+1)*self.mid_channels,...]
            channel = pooling(channel)
            channel = conv(channel)
            pools.append(channel)
        out = torch.cat(pools, dim=1)
        out = self.out_conv(out)
        return out

@MODELS.register_module()
class SE(nn.Module):
    def __init__(self, dim=1, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # squeeze操作
        y = self.fc(y).view(b, c, 1, 1) # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x) # 注意力作用每一个通道上

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