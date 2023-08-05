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
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = None,
                 init_cfg: MultiConfig = dict(type='Xavier', 
                                              layer='Conv2d',
                                              distribution='uniform'),
                 in_expand_ratio=1,
                 in_down_ratio=1,
                 mid_expand_ratio=2,
                 kernel_sizes=[1,3,3],
                 layers_num=3,
                 widen_factor=1):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = [make_divisible(in_channel, widen_factor) for in_channel in in_channels]
        self.out_channels = make_divisible(out_channels, widen_factor)

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = MSBlock(
                out_channels,
                out_channels,
                in_expand_ratio=in_expand_ratio,
                in_down_ratio = in_down_ratio,
                mid_expand_ratio=mid_expand_ratio,
                kernel_sizes=kernel_sizes,
                layers_num=layers_num,
                conv_cfg=conv_cfg, 
                act_cfg=act_cfg,
                norm_cfg=norm_cfg)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            
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