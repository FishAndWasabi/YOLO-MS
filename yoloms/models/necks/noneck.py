from typing import List, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmcv.cnn import ConvModule
from mmyolo.registry import MODELS
from mmengine.model import BaseModule
from mmdet.utils import ConfigType, MultiConfig, OptConfigType

from mmdet.models.backbones.csp_darknet import CSPLayer
@MODELS.register_module()
class NoNeck(BaseModule):
    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = None,
                 init_cfg: MultiConfig = dict(type='Xavier', 
                                              layer='Conv2d',
                                              distribution='uniform'),
                 num_csp_blocks=3,
                 deepen_factor=1,
                 widen_factor=1):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = [int(in_channel * widen_factor) for in_channel in in_channels]
        self.out_channels = int(out_channels * widen_factor)
        num_csp_blocks = round(num_csp_blocks * deepen_factor)
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channel in self.in_channels:
            l_conv = ConvModule(
                in_channel,
                self.out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = CSPLayer(
                self.out_channels,
                self.out_channels,
                num_blocks=num_csp_blocks,
                add_identity=False,
                use_depthwise=False,
                use_cspnext_block=True,
                expand_ratio=0.5,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
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