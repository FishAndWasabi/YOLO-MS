import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.cnn import Linear

from ..utils import autopad
from mmyolo.registry import MODELS
import torch.nn.functional as F


# class Partial_conv3(nn.Module):
#     def __init__(self, 
#                  dim, 
#                  kernel_size,
#                  padding,
#                  groups,
#                  conv_cfg,
#                  act_cfg,
#                  norm_cfg,
#                  n_div=3, 
#                  forward='slicing'):
#         super().__init__()
#         self.dim_conv3 = dim // n_div
#         groups = groups // n_div
#         self.dim_untouched = dim - self.dim_conv3
#         self.partial_conv3 = ConvModule(self.dim_conv3,
#                                         self.dim_conv3,
#                                         kernel_size,
#                                         padding=padding,
#                                         groups=groups,
#                                         conv_cfg=conv_cfg,
#                                         act_cfg=act_cfg,
#                                         norm_cfg=norm_cfg)
        
#         if forward == 'slicing':
#             self.forward = self.forward_slicing
#         elif forward == 'split_cat':
#             self.forward = self.forward_split_cat
#         else:
#             raise NotImplementedError

#     def forward_slicing(self, x):
#         # only for inference
#         x = x.clone()   # !!! Keep the original input intact for the residual connection later
#         x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
#         return x

#     def forward_split_cat(self, x):
#         # for training/inference
#         x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
#         x1 = self.partial_conv3(x1)
#         x = torch.cat((x1, x2), 1)
#         return x

# class E2LANLayer(nn.Module):
#     def __init__(self,
#                  in_channel,
#                  out_channel,
#                  kernel_size,
#                  conv_cfg=None,
#                  act_cfg=None,
#                  norm_cfg=None) -> None:
#         super().__init__()
#         self.in_conv = ConvModule(in_channel,
#                                   out_channel,
#                                   1,
#                                   conv_cfg=conv_cfg,
#                                   act_cfg=act_cfg,
#                                   norm_cfg=norm_cfg)        
#         self.mid_conv = Partial_conv3(out_channel,
#                                       kernel_size,
#                                       padding=autopad(kernel_size),
#                                       groups=out_channel,
#                                       conv_cfg=conv_cfg,
#                                       act_cfg=act_cfg,
#                                       norm_cfg=norm_cfg)
#         self.out_conv = ConvModule(out_channel,
#                                    in_channel,
#                                    1,
#                                    conv_cfg=conv_cfg,
#                                    act_cfg=act_cfg, 
#                                    norm_cfg=norm_cfg)
    
#     def forward(self, x):
#         x = self.in_conv(x)
#         x = self.mid_conv(x)
#         x = self.out_conv(x)
#         return x

class E2LANLayer(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 conv_cfg=None,
                 act_cfg=None,
                 norm_cfg=None) -> None:
        super().__init__()
        self.mid_conv = ConvModule(in_channel,
                                   in_channel,
                                   kernel_size,
                                   padding=autopad(kernel_size),
                                   groups=in_channel,
                                   conv_cfg=conv_cfg,
                                   act_cfg=act_cfg,
                                   norm_cfg=norm_cfg)
    
    def forward(self, x):
        # x = self.in_conv(x)
        x = self.mid_conv(x)
        # x = self.out_conv(x)
        return x


class E2LANv2(nn.Module):
    def __init__(self, 
                 in_channel,
                 out_channel,
                 in_expand_ratio=1,
                 in_down_ratio = 1,
                 mid_expand_ratio=2,
                 in_attention_cfg=None,
                 mid_attention_cfg=None,
                 out_attention_cfg=None,
                 kernel_sizes=[1,5,(3,15),(15,3)],
                 layers_num=3,
                 conv_cfg=None, 
                 act_cfg=dict(type='SiLU'),
                 norm_cfg=dict(type='BN'),
                 ) -> None:
        super().__init__()

        self.in_channel = int(in_channel*in_expand_ratio)//in_down_ratio
        self.mid_channel = self.in_channel//len(kernel_sizes)
        self.mid_expand_ratio = mid_expand_ratio
        groups = int(self.mid_channel*self.mid_expand_ratio)
        self.layers_num = layers_num
        print(self.layers_num)
        self.in_attention = None
        if in_attention_cfg is not None:
            in_attention_cfg["dim"] = in_channel
            self.in_attention = MODELS.build(in_attention_cfg)
        
        self.mid_attention = None
        if mid_attention_cfg is not None:
            mid_attention_cfg["dim"] = self.mid_channel
            self.mid_attention = MODELS.build(mid_attention_cfg)
            
        self.out_attention = None
        if out_attention_cfg is not None:
            out_attention_cfg["dim"] = out_channel
            self.out_attention = MODELS.build(out_attention_cfg)

        
        self.in_conv = ConvModule(in_channel,
                                  self.in_channel,
                                  1,
                                  conv_cfg=conv_cfg,
                                  act_cfg=act_cfg,
                                  norm_cfg=norm_cfg)
        
        self.mid_convs = []
        for kernel_size in kernel_sizes:
            if kernel_size == 1:
                self.mid_convs.append(nn.Identity())
                continue
            mid_convs = [E2LANLayer(self.mid_channel,
                                    groups,
                                    kernel_size=kernel_size,
                                    conv_cfg=conv_cfg,
                                    act_cfg=act_cfg,
                                    norm_cfg=norm_cfg) for _ in range(int(self.layers_num))]
            print("the number of layer", len(mid_convs))
            self.mid_convs.append(nn.Sequential(*mid_convs))
        self.mid_convs = nn.ModuleList(self.mid_convs)
        self.out_conv = ConvModule(self.in_channel,
                                   out_channel,
                                   1,
                                   conv_cfg=conv_cfg,
                                   act_cfg=act_cfg,
                                   norm_cfg=norm_cfg)
    
    def forward(self, x):
        out = x
        if self.in_attention is not None:
            out = self.in_attention(out)  
        out = self.in_conv(out)
        channels = []
        for i,mid_conv in enumerate(self.mid_convs):
            channel = out[:,i*self.mid_channel:(i+1)*self.mid_channel,...]
            if i >= 1:
                channel = channel + channels[i-1]
            channel = mid_conv(channel)
            channels.append(channel)
        if self.mid_attention is not None:
            channels = self.mid_attention(channels)
        out = torch.cat(channels, dim=1)
        out = self.out_conv(out)
        if self.out_attention is not None:
            out = self.out_attention(out)  
        return out