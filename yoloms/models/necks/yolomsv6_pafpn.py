import torch.nn as nn

from mmyolo.registry import MODELS
from mmyolo.models.necks.yolov6_pafpn import YOLOv6RepPAFPN
from mmyolo.models.utils import make_divisible
from mmcv.cnn import ConvModule

from ..layers import MSBlock


@MODELS.register_module()
class YOLOMSv6PAFPN(YOLOv6RepPAFPN):
    def __init__(self,
                 in_expand_ratio=1,
                 in_down_ratio=1,
                 mid_expand_ratio=2,
                 kernel_sizes=[1,3,3],
                 layers_num=3,
                 **kwargs):
        self.in_expand_ratio = in_expand_ratio
        self.in_down_ratio = in_down_ratio
        self.mid_expand_ratio = mid_expand_ratio
        self.kernel_sizes = kernel_sizes
        self.layers_num = layers_num
        super().__init__(**kwargs)
    
    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        layer0 = MSBlock(
            int((self.out_channels[idx - 1] + self.in_channels[idx - 1]) * self.widen_factor),
            int(self.out_channels[idx - 1] * self.widen_factor),
            in_expand_ratio=self.in_expand_ratio,
            in_down_ratio=self.in_down_ratio,
            mid_expand_ratio=self.mid_expand_ratio,
            kernel_sizes=self.kernel_sizes,
            layers_num=self.layers_num,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        if idx == 1:
            return layer0
        elif idx == 2:
            layer1 = ConvModule(
                in_channels=int(self.out_channels[idx - 1] *
                                self.widen_factor),
                out_channels=int(self.out_channels[idx - 2] *
                                 self.widen_factor),
                kernel_size=1,
                stride=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            return nn.Sequential(layer0, layer1)
    
    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        return MSBlock(
            int(self.out_channels[idx] * 2 * self.widen_factor),
            int(self.out_channels[idx + 1] * self.widen_factor),
            in_expand_ratio=self.in_expand_ratio,
            in_down_ratio=self.in_down_ratio,
            mid_expand_ratio=self.mid_expand_ratio,
            kernel_sizes=self.kernel_sizes,
            layers_num=self.layers_num,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
