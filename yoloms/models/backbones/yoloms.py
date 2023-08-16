# Copyright (c) MCG-NKU. All rights reserved.
import math
from typing import List, Sequence, Union

import torch.nn as nn

from mmyolo.models.backbones.base_backbone import BaseBackbone
from mmyolo.registry import MODELS

from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmcv.cnn import ConvModule

from ..layers.msblock import MSBlock


@MODELS.register_module()
class YOLOMS(BaseBackbone):
    """Backbone used in YOLOMS

    Args:
        arch (str): Architecture of YOLOMS, from {`C3-K3579'}. Defaults to `C3-K3579'.
        deepen_factor (float): Depth multiplier, multiply number of blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of channels in each layer by this amount. Defaults to 1.0.
        input_channels (int): Number of input image channels. Defaults to 3.
        out_indices (Sequence[int]): Output from which stages. Defaults to (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode). -1 means not freezing any parameters. Defaults to -1.
        plugins (list[dict]): List of plugins for stages, each dict contains:
            - cfg (dict, required): Cfg dict to build plugin.Defaults to
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
              
        in_expand_ratio (float): Channel expand ratio for inputs of MS-Block. Defaults to 3.
        mid_expand_ratio (float): Channel expand ratio for each branch in MS-Block. Defaults to 2.
        layers_num (int): Number of layer in MS-Block. Defaults to 3.
        down_ratio (float): Channel down ratio for downsample conv layer before MS-Block. Defaults to 1.
        attention_cfg (:obj:`ConfigDict` or dict, optional): Config dict for attention in MS-Block. Defaults to None.
        
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and config norm layer. Defaults to dict(type='BN').
        norm_eval (bool): Whether to set norm layers to eval mode, namely, freeze running stats (mean and var). Note: Effect on Batch Norm and its variants only.        
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer. Defaults to dict(type='SiLU', inplace=True).
        
        spp_config (:obj:`ConfigDict` or dict): Config dict for SPP Block. Defaults to dict(type="SPPFBottleneck",kernel_sizes=5).
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or list[:obj:`ConfigDict`]): Initialization config dict.
    """
    arch_settings = { 
        'C3-K3579': [[MSBlock, 80, 160,   [1, (3,3),(3,3)], False], 
                     [MSBlock, 160, 320,  [1, (5,5),(5,5)], False],
                     [MSBlock, 320, 640,  [1, (7,7),(7,7)], False], 
                     [MSBlock, 640, 1280, [1, (9,9),(9,9)], True]],
    }

    def __init__(
        self,
        arch: str = 'C3-K3579',
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        input_channels: int = 3,
        out_indices: Sequence[int] = (2, 3, 4),
        frozen_stages: int = -1,
        plugins: Union[dict, List[dict]] = None,
        
        in_expand_ratio: float = 1.,
        mid_expand_ratio: float = 2.,
        layers_num: int = 3,
        down_ratio: float = 1.,
        attention_cfg: OptConfigType = None,
        
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type='BN'),
        norm_eval: bool = False,
        act_cfg: ConfigType = dict(type='SiLU', inplace=True),
        
        spp_config: ConfigType = dict(type="SPPFBottleneck",kernel_sizes=5),
        init_cfg: OptMultiConfig = dict(
            type='Kaiming',
            layer='Conv2d',
            a=math.sqrt(5),
            distribution='uniform',
            mode='fan_in',
            nonlinearity='leaky_relu')
    ) -> None:
        arch_setting = self.arch_settings[arch]
        self.conv = ConvModule
        self.conv_cfg = conv_cfg
        
        self.in_expand_ratio = in_expand_ratio
        self.mid_expand_ratio = mid_expand_ratio
        self.attention_cfg = attention_cfg
        
        self.spp_config = spp_config
        self.down_ratio = down_ratio
        
        self.layers_num=layers_num

        super().__init__(
            arch_setting,
            deepen_factor,
            widen_factor,
            input_channels,
            out_indices,
            frozen_stages=frozen_stages,
            plugins=plugins,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            init_cfg=init_cfg)

    def build_stem_layer(self) -> nn.Module:
        """Build a stem layer."""
        stem = nn.Sequential(
            ConvModule(
                3,
                int(self.arch_setting[0][1] * self.widen_factor // 2),
                3,
                padding=1,
                stride=2,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                int(self.arch_setting[0][1] * self.widen_factor // 2),
                int(self.arch_setting[0][1] * self.widen_factor // 2),
                3,
                padding=1,
                stride=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                int(self.arch_setting[0][1] * self.widen_factor // 2),
                int(self.arch_setting[0][1] * self.widen_factor),
                3,
                padding=1,
                stride=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        return stem

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        layer, in_channels, out_channels, kernel_sizes, use_spp = setting
        
        in_channels = int(in_channels * self.widen_factor)
        out_channels = int(out_channels * self.widen_factor)

        downsample_channel = int(in_channels * self.down_ratio)
        stage = []
        conv_layer = self.conv(
            in_channels,
            downsample_channel,
            3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        stage.append(conv_layer)
        if use_spp:
            self.spp_config["in_channels"]  = downsample_channel
            self.spp_config["out_channels"]  = downsample_channel
            spp = MODELS.build(self.spp_config)
            stage.append(spp)
        
        csp_layer = layer(downsample_channel,
                          out_channels,
                          in_expand_ratio=self.in_expand_ratio,
                          in_down_ratio = 1,
                          mid_expand_ratio=self.mid_expand_ratio,
                          attention_cfg=self.attention_cfg,
                          kernel_sizes=kernel_sizes,
                          layers_num=self.layers_num * self.deepen_factor,
                          conv_cfg=self.conv_cfg, 
                          act_cfg=self.act_cfg,
                          norm_cfg=self.norm_cfg)
        stage.append(csp_layer)
        return stage