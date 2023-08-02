# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List, Sequence, Union

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from .base_backbone import BaseBackbone
from ..layers.e2lan import E2LAN
from ..layers.e2lan_nfuse import E2LAN_nfuse
from ..layers.e2lanv2 import E2LANv2
from ..layers.e2lanv3 import E2LANv3
from ..layers.e2lanv4 import E2LANv4
from ..layers.e2lanv5 import E2LANv5

@MODELS.register_module()
class YoloVX(BaseBackbone):
    arch_settings = { 
        'C4322-K3579-80-s': [[E2LAN, 80, 160,   [1, (3,3),(3,3),(3,3)], False], 
                        [E2LAN, 160, 320,  [1, (5,5),(5,5)], False],
                        [E2LAN, 320, 640,  [1, (7,7)], False], 
                        [E2LAN, 640, 1280, [1, (9,9)], True]],
        'C3322-K3579-80-s': [[E2LAN, 80, 160,   [1, (3,3),(3,3)], False], 
                        [E2LAN, 160, 320,  [1, (5,5),(5,5)], False],
                        [E2LAN, 320, 640,  [1, (7,7)], False], 
                        [E2LAN, 640, 1280, [1, (9,9)], True]],
        'C2-K3579-80': [[E2LAN, 80, 160,  [1,(3,3)], False], 
                        [E2LAN, 160, 320, [1,(5,5)], False],
                        [E2LAN, 320, 640, [1,(7,7)], False], 
                        [E2LAN, 640, 1280,[1,(9,9)], True]],
        'C4-K3579-80': [[E2LAN, 80, 160,  [1, 3, (3,3),(3,3)], False], 
                        [E2LAN, 160, 320, [1, 3, (5,5),(5,5)], False],
                        [E2LAN, 320, 640, [1 ,3, (7,7),(7,7)], False], 
                        [E2LAN, 640, 1280,[1, 3, (9,9),(9,9)], True]],
        'C5-K3579-80': [[E2LAN, 80, 160,   [1, 3, (3,3),(3,3),(3,3)], False], 
                        [E2LAN, 160, 320,  [1, 3, (5,5),(5,5),(5,5)], False],
                        [E2LAN, 320, 640,  [1 , 3, (7,7),(7,7),(7,7)], False], 
                        [E2LAN, 640, 1280, [1, 3, (9,9),(9,9),(9,9)], True]],
        'C3-K3579-80': [[E2LAN, 80, 160, [1, (3,3),(3,3)], False], 
                        [E2LAN, 160, 320,  [1, (3,5),(5,3)], False],
                        [E2LAN, 320, 640,  [1 , (3,7),(7,3)], False], 
                        [E2LAN, 640, 1280, [1, (3,9),(9,3)], True]],
        'C3-K3579-80v5': [[E2LANv5, 80, 160, [1, (3,3),(5,5)], False], 
                        [E2LANv5, 160, 320,  [1, (5,5),(7,7)], False],
                        [E2LANv5, 320, 640,  [1 , (7,7),(9,9)], False], 
                        [E2LANv5, 640, 1280, [1, (9,9),(11,11)], True]],
        'C3-K3579-80v2': [[E2LANv2, 80, 160, [1, (3,3),(3,3)], False], 
                        [E2LANv2, 160, 320,  [1, (3,5),(5,3)], False],
                        [E2LANv2, 320, 640,  [1 , (3,7),(7,3)], False], 
                        [E2LANv2, 640, 1280, [1, (3,9),(9,3)], True]],
        'C3-K3579-80v3': [[E2LANv3, 80, 160, [1, (3,3),(3,3)], False], 
                        [E2LANv3, 160, 320,  [1, (3,5),(5,3)], False],
                        [E2LANv3, 320, 640,  [1 , (3,7),(7,3)], False], 
                        [E2LANv3, 640, 1280, [1, (3,9),(9,3)], True]],
        'C3-K3579-80v4': [[E2LANv4, 60, 160, [1, (3,3),(3,3)], False], 
                        [E2LANv4, 160, 320,  [1, (3,5),(5,3)], False],
                        [E2LANv4, 320, 640,  [1 , (3,7),(7,3)], False], 
                        [E2LANv4, 640, 1280, [1, (3,9),(9,3)], True]],
        'C3-K3357-80v4': [[E2LANv4, 60, 120, [1, (3,3),(3,3)], False], 
                        [E2LANv4, 120, 320,  [1, (3,3),(3,3)], False],
                        [E2LANv4, 320, 640,  [1 , (3,5),(5,3)], False], 
                        [E2LANv4, 640, 1280, [1, (3,7),(7,3)], True]],
        'C3-K3557-80v4': [[E2LANv4, 60, 120, [1, (3,3),(3,3)], False], 
                        [E2LANv4, 120, 320,  [1, (3,5),(5,3)], False],
                        [E2LANv4, 320, 640,  [1 , (3,5),(5,3)], False], 
                        [E2LANv4, 640, 1280, [1, (3,7),(7,3)], True]],
        'C3-K3333-80v4': [[E2LANv4, 60, 120, [1, (3,3),(3,3)], False], 
                        [E2LANv4, 120, 240,  [1, (3,3),(3,3)], False],
                        [E2LANv4, 240, 480,  [1 , (3,3),(3,3)], False], 
                        [E2LANv4, 480, 960, [1, (3,3),(3,3)], True]],
        'C3-K3579-80-s': [[E2LAN, 80, 160,   [1, (3,3),(3,3)], False], 
                        [E2LAN, 160, 320,  [1, (5,5),(5,5)], False],
                        [E2LAN, 320, 640,  [1, (7,7),(7,7)], False], 
                        [E2LAN, 640, 1280, [1, (9,9),(9,9)], True]],
        'C3-K3579-80-s-ncp': [[E2LAN, 80, 160, [(3,3),(3,3)], False], 
                        [E2LAN, 160, 320,  [(5,5),(5,5)], False],
                        [E2LAN, 320, 640,  [(7,7),(7,7)], False], 
                        [E2LAN, 640, 1280, [(9,9),(9,9)], True]],
        'C3-K3579-80-s-nfuse': [[E2LAN_nfuse, 80, 160,   [1, (3,3),(3,3)], False], 
                        [E2LAN_nfuse, 160, 320,  [1, (5,5),(5,5)], False],
                        [E2LAN_nfuse, 320, 640,  [1, (7,7),(7,7)], False], 
                        [E2LAN_nfuse, 640, 1280, [1, (9,9),(9,9)], True]],
        'C3-K3333-80': [[E2LAN, 80, 160,   [1, (3,3),(3,3)], False], 
                        [E2LAN, 160, 320,  [1, (3,3),(3,3)], False],
                        [E2LAN, 320, 640,  [1, (3,3),(3,3)], False], 
                        [E2LAN, 640, 1280, [1, (3,3),(3,3)], True]],
        'C3-K5555-80': [[E2LAN, 80, 160,   [1, (5,5),(5,5)], False], 
                        [E2LAN, 160, 320,  [1, (5,5),(5,5)], False],
                        [E2LAN, 320, 640,  [1, (5,5),(5,5)], False], 
                        [E2LAN, 640, 1280, [1, (5,5),(5,5)], True]],
        'C3-K7777-80': [[E2LAN, 80, 160,   [1, (7,7),(7,7)], False], 
                        [E2LAN, 160, 320,  [1, (7,7),(7,7)], False],
                        [E2LAN, 320, 640,  [1, (7,7),(7,7)], False], 
                        [E2LAN, 640, 1280, [1, (7,7),(7,7)], True]],
        'C3-K9999-80': [[E2LAN, 80, 160,   [1, (9,9),(9,9)], False], 
                        [E2LAN, 160, 320,  [1, (9,9),(9,9)], False],
                        [E2LAN, 320, 640,  [1, (9,9),(9,9)], False], 
                        [E2LAN, 640, 1280, [1, (9,9),(9,9)], True]],
        'C3-K9753-80': [[E2LAN, 80, 160,   [1, (9,9),(9,9)], False], 
                        [E2LAN, 160, 320,  [1, (7,7),(7,7)], False],
                        [E2LAN, 320, 640,  [1, (5,5),(5,5)], False], 
                        [E2LAN, 640, 1280, [1, (3,3),(3,3)], True]],
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
        norm_eval: bool = False,
        in_expand_ratio=1,
        mid_expand_ratio=2,
        layers_num=3,
        in_attention_cfg=None,
        mid_attention_cfg=None,
        out_attention_cfg=None,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type='BN'),
        act_cfg: ConfigType = dict(type='SiLU', inplace=True),
        spp_config = dict(type="SPPFBottleneck",kernel_sizes=5),
        down_ratio = 1,
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
        self.in_attention_cfg = in_attention_cfg
        self.mid_attention_cfg = mid_attention_cfg
        self.out_attention_cfg = out_attention_cfg
        
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
        

        # print("backbone",downsample_channel, out_channels)
        csp_layer = layer(downsample_channel,
                          out_channels,
                          in_expand_ratio=self.in_expand_ratio,
                          in_down_ratio = 1,
                          mid_expand_ratio=self.mid_expand_ratio,
                          in_attention_cfg=self.in_attention_cfg,
                          mid_attention_cfg=self.mid_attention_cfg,
                          out_attention_cfg=self.out_attention_cfg,
                          kernel_sizes=kernel_sizes,
                          layers_num=self.layers_num * self.deepen_factor,
                          conv_cfg=self.conv_cfg, 
                          act_cfg=self.act_cfg,
                          norm_cfg=self.norm_cfg)
        stage.append(csp_layer)
        return stage