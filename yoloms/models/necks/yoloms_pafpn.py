# Copyright (c) MCG-NKU. All rights reserved.
import math
from typing import Sequence, Union

import torch.nn as nn

from mmyolo.models.necks.base_yolo_neck import BaseYOLONeck
from mmyolo.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from ..layers.msblock import MSBlock


@MODELS.register_module()
class YOLOMSPAFPN(BaseYOLONeck):
    """Path Aggregation Network with MS-Blocks.

    Args:
        in_channels (Sequence[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        mid_channels (Sequence[int]): Number of middle channels per scale. Defaults to [].
        
        deepen_factor (float): Depth multiplier, multiply number of blocks in MS-Block layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of channels in each layer by this amount. Defaults to 1.0.
        freeze_all(bool): Whether to freeze the model. Defaults to False.
        use_depthwise (bool): Whether to use depthwise separable convolution in blocks. Defaults to False.
        upsample_cfg (dict): Config dict for interpolate layer. Default: `dict(scale_factor=2, mode='nearest')`.
        
        in_expand_ratio (float): Channel expand ratio for inputs of MS-Block. Defaults to 3.
        in_down_ratio (float): Channel down ratio for downsample conv layer in MS-Block. Defaults to 1.
        mid_expand_ratio (float): Channel expand ratio for each branch in MS-Block. Defaults to 2.
        layers_num (int): Number of layer in MS-Block. Defaults to 3.
        kernel_sizes (list(int, tuple[int])): Sequential or number of kernel sizes in MS-Block. Defaults to [1,3,3].
        attention_cfg (:obj:`ConfigDict` or dict, optional): Config dict for attention in MS-Block. Defaults to None.
        
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and config norm layer. Defaults to dict(type='BN').
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer. Defaults to dict(type='SiLU', inplace=True).
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or list[:obj:`ConfigDict`]): Initialization config dict.
    """
    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        mid_channels: Sequence[int] = [],
        
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        freeze_all: bool = False,
        use_depthwise: bool = False,
        upsample_cfg: ConfigType = dict(scale_factor=2, mode='nearest'),
        
        in_expand_ratio: float = 1,
        in_down_ratio: float = 1,
        mid_expand_ratio: float = 2,
        layers_num: int = 3,
        kernel_sizes: Sequence[Union[int, Sequence[int]]] = [1,3,3],
        attention_cfg: OptConfigType = None,
        
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
            attention_cfg=attention_cfg,
            kernel_sizes=kernel_sizes,
            layers_num= round(layers_num * deepen_factor),
        )          
        
        self.conv_cfg = conv_cfg
        self.mid_channels = [int(mid_channel * widen_factor) for mid_channel in mid_channels]
        
        self.layer = MSBlock
        
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
        if self.in_channels[idx] != self.mid_channels[idx]:
            proj = self.conv(self.in_channels[idx],
                            self.mid_channels[idx],
                            1,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg)
        else:
            proj = nn.Identity()
        if idx == len(self.in_channels) - 1:
            layer = self.conv(
                self.mid_channels[idx],
                self.mid_channels[idx - 1],
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
            return self.layer(self.mid_channels[idx - 1] * 2,
                              self.mid_channels[idx - 1],**self.layer_config)
        else:
            return nn.Sequential(
                self.layer(self.mid_channels[idx - 1] * 2,
                           self.mid_channels[idx - 1],
                           **self.layer_config),
                self.conv(self.mid_channels[idx - 1],
                          self.mid_channels[idx - 2],
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
        return self.conv(self.mid_channels[idx],
                         self.mid_channels[idx],
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
        return self.layer(self.mid_channels[idx] * 2,
                          self.mid_channels[idx + 1],
                          **self.layer_config)

    def build_out_layer(self, idx: int) -> nn.Module:
        """build out layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The out layer.
        """
        return self.conv(
            self.mid_channels[idx],
            self.out_channels,
            3,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)



