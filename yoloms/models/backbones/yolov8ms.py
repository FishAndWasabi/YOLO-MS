# Copyright (c) MCG-NKU. All rights reserved.
import torch
import torch.nn as nn

from mmyolo.models.backbones.base_backbone import BaseBackbone
from mmyolo.models.utils import make_divisible
from mmyolo.registry import MODELS

from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptConfigType

from ..layers.msblock import MSBlock


@MODELS.register_module()
class YOLOv8MS(BaseBackbone):
    """Backbone used in YOLOv8-MS

    Args:
        arch (str): Architecture of YOLOMS, from {`C3-K3579'}. Defaults to `C3-K3579'.
        last_stage_out_channels (int): Final layer output channel. Defaults to 1024.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for convolution layer. Defaults to None.
            
        in_expand_ratio (float): Channel expand ratio for inputs of MS-Block. Defaults to 3.
        mid_expand_ratio (float): Channel expand ratio for each branch in MS-Block. Defaults to 2.
        layers_num (int): Number of layer in MS-Block. Defaults to 3.
        
        attention_cfg (:obj:`ConfigDict` or dict, optional): Config dict for attention in MS-Block. Defaults to None.
        
        spp_config (:obj:`ConfigDict` or dict, optional): Config dict for SPP Block. Defaults to dict(type="SPPFBottleneck",kernel_sizes=5).
    """
    arch_settings = { 
        'C3-K3579': [[MSBlock, 80, 160,   [1, (3,3),(3,3)], False], 
                     [MSBlock, 160, 320,  [1, (5,5),(5,5)], False],
                     [MSBlock, 320, 640,  [1, (7,7),(7,7)], False], 
                     [MSBlock, 640, None, [1, (9,9),(9,9)], True]],
        'C4-K3579': [[MSBlock, 64, 128,   [1, (3,3),(3,3)], False], 
                    [MSBlock, 128, 256,  [1, (5,5),(5,5)], False],
                    [MSBlock, 256, 512,  [1, (7,7),(7,7)], False], 
                    [MSBlock, 512, None, [1, (9,9),(9,9)], True]]
    }

    def __init__(self,
                 arch: str = 'C3-K3579',
                 last_stage_out_channels: int = 1024,
                 conv_cfg: OptConfigType = None,
                 in_expand_ratio: float = 1.,
                 mid_expand_ratio: float = 2.,
                 layers_num: int = 3,
                 attention_cfg: OptConfigType = None,
                 spp_config: ConfigType = dict(type="SPPFBottleneck",kernel_sizes=5),
                 **kwargs):
        self.arch_settings[arch][-1][2] = last_stage_out_channels
        self.conv = ConvModule
        self.conv_cfg = conv_cfg
        
        self.in_expand_ratio = in_expand_ratio
        self.mid_expand_ratio = mid_expand_ratio
        self.attention_cfg = attention_cfg
        
        self.spp_config = spp_config
        
        self.layers_num=layers_num
        
        super().__init__(self.arch_settings[arch], 
                         **kwargs)

    def build_stem_layer(self) -> nn.Module:
        """Build a stem layer."""
        return ConvModule(
            self.input_channels,
            make_divisible(self.arch_setting[0][1], self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
    
    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        layer, in_channels, out_channels, kernel_sizes, use_spp = setting
        
        in_channels = make_divisible(in_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)

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
    
    def init_weights(self):
        """Initialize the parameters."""
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    # In order to be consistent with the source code,
                    # reset the Conv2d initialization parameters
                    m.reset_parameters()
        else:
            super().init_weights()