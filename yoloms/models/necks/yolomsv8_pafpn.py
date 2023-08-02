import torch.nn as nn

from mmyolo.registry import MODELS
from mmyolo.models.necks.yolov8_pafpn import YOLOv8PAFPN
from mmyolo.models.utils import make_divisible

from ..layers import MSBlock


@MODELS.register_module()
class YOLOMSv8PAFPN(YOLOv8PAFPN):
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
        return MSBlock(
            make_divisible((self.in_channels[idx - 1] + self.in_channels[idx]),
                           self.widen_factor),
            make_divisible(self.out_channels[idx - 1], self.widen_factor),
            in_expand_ratio=self.in_expand_ratio,
            in_down_ratio=self.in_down_ratio,
            mid_expand_ratio=self.mid_expand_ratio,
            kernel_sizes=self.kernel_sizes,
            layers_num=self.layers_num,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
    
    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        return MSBlock(
            make_divisible(
                (self.out_channels[idx] + self.out_channels[idx + 1]),
                self.widen_factor),
            make_divisible(self.out_channels[idx + 1], self.widen_factor),
            in_expand_ratio=self.in_expand_ratio,
            in_down_ratio=self.in_down_ratio,
            mid_expand_ratio=self.mid_expand_ratio,
            kernel_sizes=self.kernel_sizes,
            layers_num=self.layers_num,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
