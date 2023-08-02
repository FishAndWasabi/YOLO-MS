# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Sequence

import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS

from ..layers.msblock import MSBlock

from mmyolo.models.necks.base_yolo_neck import BaseYOLONeck

@MODELS.register_module()
class YoloMSPAFPN(BaseYOLONeck):
    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        feat_channels: Sequence[int] = [],
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        freeze_all: bool = False,
        use_depthwise: bool = False,
        upsample_cfg: ConfigType = dict(scale_factor=2, mode='nearest'),
        
        in_expand_ratio=1,
        in_down_ratio = 1,
        mid_expand_ratio=2,
        in_attention_cfg=None,
        mid_attention_cfg=None,
        out_attention_cfg=None,
        kernel_sizes=[1,5,(3,15),(15,3)],
        layers_num: int = 3,
        
        layer_type = "msblock",
        conv_cfg: bool = None,
        norm_cfg: ConfigType = dict(type='BN'),
        act_cfg: ConfigType = dict(type='SiLU', inplace=True),
        init_cfg: OptMultiConfig = dict(
            type='Kaiming',
            layer='Conv2d',
            a=math.sqrt(5),
            distribution='uniform',
            mode='fan_in',
            nonlinearity='leaky_relu')
    ) -> None:
        
        self.conv = DepthwiseSeparableConvModule \
            if use_depthwise else ConvModule
        self.upsample_cfg = upsample_cfg
        
        self.layer_config = dict(
            in_expand_ratio=in_expand_ratio,
            in_down_ratio = in_down_ratio,
            mid_expand_ratio=mid_expand_ratio,
            in_attention_cfg=in_attention_cfg,
            mid_attention_cfg=mid_attention_cfg,
            out_attention_cfg=out_attention_cfg,
            kernel_sizes=kernel_sizes,
            layers_num= round(layers_num * deepen_factor),
        )          
        
        self.conv_cfg = conv_cfg
        self.feat_channels = [int(feat_channel*widen_factor) for feat_channel in feat_channels]
        
        self.layer = {"msblock": MSBlock}[layer_type]
        
        super().__init__(
            in_channels=[
                int(channel * widen_factor) for channel in in_channels
            ],
            out_channels=int(out_channels * widen_factor),
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        """
        if self.in_channels[idx] != self.feat_channels[idx]:
            proj = self.conv(self.in_channels[idx],
                            self.feat_channels[idx],
                            1,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg)
        else:
            proj = nn.Identity()
        if idx == len(self.in_channels) - 1:
            layer = self.conv(
                self.feat_channels[idx],
                self.feat_channels[idx - 1],
                1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            layer = nn.Identity()

        return nn.Sequential(proj, layer)

    def build_upsample_layer(self, *args, **kwargs) -> nn.Module:
        """build upsample layer."""
        return nn.Upsample(**self.upsample_cfg)

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        if idx == 1:
            return self.layer(self.feat_channels[idx - 1] * 2,
                              self.feat_channels[idx - 1],**self.layer_config)
        else:
            return nn.Sequential(
                self.layer(self.feat_channels[idx - 1] * 2,
                           self.feat_channels[idx - 1],
                           **self.layer_config),
                self.conv(self.feat_channels[idx - 1],
                          self.feat_channels[idx - 2],
                          kernel_size=1,
                          norm_cfg=self.norm_cfg,
                          act_cfg=self.act_cfg))

    def build_downsample_layer(self, idx: int) -> nn.Module:
        """build downsample layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The downsample layer.
        """
        return self.conv(self.feat_channels[idx],
                         self.feat_channels[idx],
                         kernel_size=3,
                         stride=2,
                         padding=1,
                         norm_cfg=self.norm_cfg,
                         act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        return self.layer(self.feat_channels[idx] * 2,
                          self.feat_channels[idx + 1],
                          **self.layer_config)

    def build_out_layer(self, idx: int) -> nn.Module:
        """build out layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The out layer.
        """
        return self.conv(
            self.feat_channels[idx],
            self.out_channels,
            3,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)



