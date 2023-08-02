# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List, Sequence, Union

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from .base_backbone import BaseBackbone
from ..layers import E2LAN, E2LAN_SK, E2LAN_nfuse

@MODELS.register_module()
class E2LANNeXt(BaseBackbone):
    """CSPNeXt backbone used in RTMDet.

    Args:
        arch (str): Architecture of CSPNeXt, from {P5, P6}.
            Defaults to P5.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Defaults to (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Defaults to -1.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Defaults to False.
        arch_ovewrite (list): Overwrite default arch settings.
            Defaults to None.
        channel_attention (bool): Whether to add channel attention in each
            stage. Defaults to True.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and
            config norm layer. Defaults to dict(type='BN', requires_grad=True).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to dict(type='SiLU').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`]): Initialization config dict.
    """
    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    arch_settings = { 
        'C3333-K3579': [[E2LAN, 64, 128, 3, [1, 3, (3,3),(3,3)], False], 
                        [E2LAN, 128, 256, 3,  [1, 3, (3,5),(5,3)], False],
                        [E2LAN, 256, 512, 3, [1 ,3, (3,7),(7,3)], False], 
                        [E2LAN, 512, 1024, 3, [1, 3, (3,9),(9,3)], True]],
        
        'C2222-K3579': [[E2LAN, 64, 128, 2, [1, 3, (3,3),(3,3)], False], 
                        [E2LAN, 128, 256, 2,  [1, 3, (3,5),(5,3)], False],
                        [E2LAN, 256, 512, 2, [1 ,3, (3,7),(7,3)], False], 
                        [E2LAN, 512, 1024, 2, [1, 3, (3,9),(9,3)], True]],
        'C3333-K3579-80': [[E2LAN, 80, 160, 3, [1, 3, (3,3),(3,3)], False], 
                        [E2LAN, 160, 320, 3,  [1, 3, (3,5),(5,3)], False],
                        [E2LAN, 320, 640, 3, [1 , 3, (3,7),(7,3)], False], 
                        [E2LAN, 640, 1280, 3, [1, 3, (3,9),(9,3)], True]],
        'C33333-K3579-80': [[E2LAN, 80, 160, 3, [1, 3, (3,3),(3,3),(3,3)], False], 
                        [E2LAN, 160, 320, 3,  [1, 3, (3,5),(5,3),(5,5)], False],
                        [E2LAN, 320, 640, 3, [1 , 3, (3,7),(7,3),(7,7)], False], 
                        [E2LAN, 640, 1280, 3, [1, 3, (3,9),(9,3),(9,9)], True]],
        'C333-K3579-80': [[E2LAN, 80, 160, 3, [1, (3,3),(3,3)], False], 
                        [E2LAN, 160, 320, 3,  [1, (3,5),(5,3)], False],
                        [E2LAN, 320, 640, 3, [1 , (3,7),(7,3)], False], 
                        [E2LAN, 640, 1280, 3, [1, (3,9),(9,3)], True]],
        'C222-K3579-80': [[E2LAN, 80, 160, 2, [1, (3,3),(3,3)], False], 
                        [E2LAN, 160, 320, 2,  [1, (3,5),(5,3)], False],
                        [E2LAN, 320, 640, 2, [1 , (3,7),(7,3)], False], 
                        [E2LAN, 640, 1280, 2, [1, (3,9),(9,3)], True]],
        'C2222-K3579-80': [[E2LAN, 80, 160, 2, [1,(3,3), (3,3),(3,3)], False], 
                            [E2LAN, 160, 320, 2,  [1,(3,3), (3,5),(5,3)], False],
                            [E2LAN, 320, 640, 2, [1 ,(3,3), (3,7),(7,3)], False], 
                            [E2LAN, 640, 1280, 2, [1,(3,3), (3,9),(9,3)], True]],
        'C222-K3579-80+SK': [[E2LAN_SK, 80, 160, 2, [1, (3,3),(3,3)], False], 
                        [E2LAN_SK, 160, 320, 2,  [1, (3,5),(5,3)], False],
                        [E2LAN_SK, 320, 640, 2, [1 , (3,7),(7,3)], False], 
                        [E2LAN_SK, 640, 1280, 2, [1, (3,9),(9,3)], True]],
        
        'C3355-K3579': [[E2LAN, 64, 128, 3, [1, 3, (3,3),(3,3)], False], 
                        [E2LAN, 128, 256, 3,  [1, 3, (3,5),(5,3)], False],
                        [E2LAN, 256, 512, 5, [1 ,3, (3,7),(7,3)], False], 
                        [E2LAN, 512, 1024, 5, [1, 3, (3,9),(9,3)], True]],
        
        'C3333-K3333': [[E2LAN, 64, 128, 3, [1, 3, (3,3),(3,3)], False], 
                        [E2LAN, 128, 256, 3,  [1, 3, (3,3),(3,3)], False],
                        [E2LAN, 256, 512, 3, [1 ,3, (3,3),(3,3)], False], 
                        [E2LAN, 512, 1024, 3, [1, 3, (3,3),(3,3)], True]],
        
        'C4444-K3333': [[E2LAN, 64, 128, 3, [1, 3, (3,3),(3,3)], False], 
                        [E2LAN, 128, 256, 3,  [1, 3, (3,3),(3,3)], False],
                        [E2LAN, 256, 512, 3, [1 ,3, (3,3),(3,3)], False], 
                        [E2LAN, 512, 1024, 3, [1, 3, (3,3),(3,3)], True]],
        
        'C3333-K5555': [[E2LAN, 64, 128, 3, [1, 3, (3,5),(5,3)], False], 
                        [E2LAN, 128, 256, 3,  [1, 3, (3,5),(5,3)], False],
                        [E2LAN, 256, 512, 3, [1 ,3, (3,5),(5,3)], False], 
                        [E2LAN, 512, 1024, 3, [1, 3, (3,5),(5,3)], True]],
        
        'C3333-K7777': [[E2LAN, 64, 128, 3, [1, 3, (3,7),(7,3)], False], 
                        [E2LAN, 128, 256, 3,  [1, 3, (3,7),(7,3)], False],
                        [E2LAN, 256, 512, 3, [1 ,3, (3,7),(7,3)], False], 
                        [E2LAN, 512, 1024, 3, [1, 3, (3,7),(7,3)], True]],
        
        'C1357-K3579': [[E2LAN, 64, 128, 1, [1, 3, (3,3),(3,3)], False], 
                        [E2LAN, 128, 256, 3,  [1, 3, (3,5),(5,3)], False],
                        [E2LAN, 256, 512, 5, [1 ,3, (3,7),(7,3)], False], 
                        [E2LAN, 512, 1024, 7, [1, 3, (3,9),(9,3)], True]],
        
        'C5555-K3579': [[E2LAN, 64, 128, 5, [1, 3, (3,3),(3,3)], False], 
                        [E2LAN, 128, 256, 5,  [1, 3, (3,5),(5,3)], False],
                        [E2LAN, 256, 512, 5, [1 ,3, (3,7),(7,3)], False], 
                        [E2LAN, 512, 1024, 5, [1, 3, (3,9),(9,3)], True]],
    }

    def __init__(
        self,
        arch: str = 'C3333-K3579',
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        input_channels: int = 3,
        out_indices: Sequence[int] = (2, 3, 4),
        frozen_stages: int = -1,
        plugins: Union[dict, List[dict]] = None,
        expand_ratio: float = 0.5,
        downsample_ratio: float = 2, 
        first_expand_ratio: float=1,
        arch_ovewrite: dict = None,
        attention_cfg = None,
        mid_attention_cfg = None,
        back_attention_cfg = None,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type='BN'),
        act_cfg: ConfigType = dict(type='SiLU', inplace=True),
        norm_eval: bool = False,
        use_ln=False,
        spp_config = dict(type="SPPFBottleneck",kernel_sizes=5),
        init_cfg: OptMultiConfig = dict(
            type='Kaiming',
            layer='Conv2d',
            a=math.sqrt(5),
            distribution='uniform',
            mode='fan_in',
            nonlinearity='leaky_relu')
    ) -> None:
        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        self.conv = ConvModule
        self.expand_ratio = expand_ratio
        self.downsample_ratio = downsample_ratio
        self.first_expand_ratio = first_expand_ratio
        self.conv_cfg = conv_cfg
        self.use_ln = use_ln
        
        self.attention_cfg = attention_cfg
        self.mid_attention_cfg = mid_attention_cfg
        self.back_attention_cfg = back_attention_cfg
        self.spp_config = spp_config
        

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
        layer, in_channels, out_channels, expand_ratio, kernel_sizes, use_spp = setting

        in_channels = int(in_channels * self.widen_factor)
        out_channels = int(out_channels * self.widen_factor)

        
        downsample_channel = int(in_channels * self.downsample_ratio)
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
        

        # print("backbone",downsample_channel, out_channels)
        csp_layer = layer(downsample_channel,
                          out_channels,
                          expand_ratio=expand_ratio,
                          conv_cfg=None,
                          norm_cfg=self.norm_cfg,
                          act_cfg=self.act_cfg,
                          kernel_sizes=kernel_sizes,
                          use_ln=self.use_ln,
                          mid_attention_cfg = self.mid_attention_cfg,
                          first_expand_ratio=self.first_expand_ratio,
                          back_attention_cfg=self.back_attention_cfg,
                          attention_cfg=self.attention_cfg)
        stage.append(csp_layer)
        return stage