# Copyright (c) MCG-NKU. All rights reserved.
import torch.nn as nn

from mmyolo.models.backbones.base_backbone import BaseBackbone
from mmyolo.registry import MODELS

from mmdet.utils import ConfigType, OptConfigType
from mmcv.cnn import ConvModule

from ..layers.msblock import MSBlock


@MODELS.register_module()
class YOLOv6MS(BaseBackbone):
    """Backbone used in YOLOv6-MS

    Args:
        arch (str): Architecture of YOLOMS, from {`C3-K3579'}. Defaults to `C3-K3579'.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for convolution layer. Defaults to None.
            
        in_expand_ratio (float): Channel expand ratio for inputs of MS-Block. Defaults to 3.
        mid_expand_ratio (float): Channel expand ratio for each branch in MS-Block. Defaults to 2.
        layers_num (int): Number of layer in MS-Block. Defaults to 3.
        
        attention_cfg (:obj:`ConfigDict` or dict, optional): Config dict for attention in MS-Block. Defaults to None.
        
        spp_config (:obj:`ConfigDict` or dict, optional): Config dict for SPP Block. Defaults to dict(type="SPPFBottleneck",kernel_sizes=5).
        block_cfg (dict): Config dict for the block used to build each layer. Defaults to dict(type='RepVGGBlock').
    """
    arch_settings = { 
        'C3-K3579': [[MSBlock, 80, 160,   [1, (3,3),(3,3)], False], 
                     [MSBlock, 160, 320,  [1, (5,5),(5,5)], False],
                     [MSBlock, 320, 640,  [1, (7,7),(7,7)], False], 
                     [MSBlock, 640, 1280, [1, (9,9),(9,9)], True]],
    }

    def __init__(self,
                 arch: str = 'C3-K3579',
                 conv_cfg: OptConfigType = None,
                 in_expand_ratio: float = 1.,
                 mid_expand_ratio: float = 2.,
                 layers_num: int = 3,
                 attention_cfg: OptConfigType = None,
                 spp_config: ConfigType = dict(type="SPPFBottleneck",kernel_sizes=5),
                 block_cfg: ConfigType = dict(type='RepVGGBlock'),
                 **kwargs):
        self.conv = ConvModule
        self.conv_cfg = conv_cfg
        
        self.in_expand_ratio = in_expand_ratio
        self.mid_expand_ratio = mid_expand_ratio
        self.attention_cfg = attention_cfg
        
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
                          attention_cfg=self.attention_cfg,
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