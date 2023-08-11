# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmcv.cnn import ConvModule
from mmdet.utils import OptConfigType

from mmyolo.registry import MODELS
from ..layers.msblock import MSBlock
from mmyolo.models.backbones.base_backbone import BaseBackbone
from mmyolo.models.backbones.efficient_rep import YOLOv6CSPBep
from mmyolo.models.utils import make_divisible, make_round
from mmdet.utils import ConfigType, OptMultiConfig

@MODELS.register_module()
class YOLOMSv6(BaseBackbone):
    arch_settings = { 
        'C3-K3579': [[MSBlock, 80, 160,   [1, (3,3),(3,3)], False], 
                     [MSBlock, 160, 320,  [1, (5,5),(5,5)], False],
                     [MSBlock, 320, 640,  [1, (7,7),(7,7)], False], 
                     [MSBlock, 640, 1280, [1, (9,9),(9,9)], True]],
    }

    def __init__(self,
                 arch: str = 'C3-K3579',
                 conv_cfg: OptConfigType = None,
                 in_expand_ratio=3,
                 mid_expand_ratio=2,
                 layers_num=1,
                 in_attention_cfg=None,
                 mid_attention_cfg=None,
                 out_attention_cfg=None,
                 spp_config = dict(type="SPPFBottleneck",kernel_sizes=5),
                 block_cfg: ConfigType = dict(type='RepVGGBlock'),
                 **kwargs):
        self.conv = ConvModule
        self.conv_cfg = conv_cfg
        
        self.in_expand_ratio = in_expand_ratio
        self.mid_expand_ratio = mid_expand_ratio
        self.in_attention_cfg = in_attention_cfg
        self.mid_attention_cfg = mid_attention_cfg
        self.out_attention_cfg = out_attention_cfg
        
        self.spp_config = spp_config
        
        self.layers_num=layers_num
        self.block_cfg = block_cfg
        super().__init__(self.arch_settings[arch], 
                         **kwargs)
    
    def build_stem_layer(self) -> nn.Module:
        """Build a stem layer."""

        block_cfg = self.block_cfg.copy()
        block_cfg.update(
            dict(
                in_channels=self.input_channels,
                out_channels=int(self.arch_setting[0][1] * self.widen_factor),
                kernel_size=3,
                stride=2,
            ))
        return MODELS.build(block_cfg)
    
    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        layer, in_channels, out_channels, kernel_sizes, use_spp = setting
        
        in_channels = int(in_channels * self.widen_factor)
        out_channels = int(out_channels * self.widen_factor)

        stage = []
        conv_layer = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        stage.append(conv_layer)
        csp_layer = layer(out_channels,
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
        if use_spp:
            self.spp_config["in_channels"]  = out_channels
            self.spp_config["out_channels"]  = out_channels
            spp = MODELS.build(self.spp_config)
            stage.append(spp)
        return stage