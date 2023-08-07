from typing import List, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmcv.cnn import ConvModule
from mmyolo.registry import MODELS
from mmengine.model import BaseModule
from mmdet.utils import ConfigType, MultiConfig, OptConfigType
from mmyolo.models.utils import make_divisible

from ..layers.msblock import MSBlock

@MODELS.register_module()
class YOLOMSNeck(BaseModule):
    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 feat_channels,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = None,
                 init_cfg: MultiConfig = dict(type='Xavier', 
                                              layer='Conv2d',
                                              distribution='uniform'),
                 in_expand_ratio=3,
                 in_down_ratio=1,
                 mid_expand_ratio=2,
                 kernel_sizes=[1,3,3],
                 layers_num=3,
                 widen_factor=1):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = [int(in_channel * widen_factor) for in_channel in in_channels]
        self.feat_channels = [int(feat_channel * widen_factor) for feat_channel in feat_channels]
        self.out_channels = int(out_channels * widen_factor)

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channel, feat_channel in zip(self.in_channels,
                                            self.feat_channels):
            l_conv = ConvModule(
                in_channel,
                feat_channel,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = MSBlock(
                feat_channel,
                self.out_channels,
                in_expand_ratio=in_expand_ratio,
                in_down_ratio = in_down_ratio,
                mid_expand_ratio=mid_expand_ratio,
                kernel_sizes=kernel_sizes,
                layers_num=layers_num,
                conv_cfg=conv_cfg, 
                act_cfg=act_cfg,
                norm_cfg=norm_cfg)
            # fpn_conv = ConvModule(
            #     feat_channel,
            #     self.out_channels,
            #     3,
            #     padding=2,
            #     conv_cfg=conv_cfg,
            #     norm_cfg=norm_cfg,
            #     act_cfg=act_cfg,
            #     inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(nn.Sequential(fpn_conv))
            
            
    def forward(self, inputs: Tuple[Tensor]) -> tuple:

        assert len(inputs) == len(self.in_channels)

        # build laterals
        outs = [
            lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        outs = [
            fpn_conv(outs[i]) for i, fpn_conv in enumerate(self.fpn_convs)
        ]
        
        return tuple(outs)