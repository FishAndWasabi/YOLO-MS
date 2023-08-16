# Copyright (c) MCG-NKU. All rights reserved.
from typing import Sequence, Union

import torch.nn as nn

from mmyolo.registry import MODELS
from mmyolo.models.necks.yolov8_pafpn import YOLOv8PAFPN
from mmyolo.models.utils import make_divisible

from ..layers import MSBlock


@MODELS.register_module()
class YOLOv8MSPAFPN(YOLOv8PAFPN):
    """Path Aggregation Network in YOLOv8 with MS-Blocks.

    Args:
        in_expand_ratio (float): Channel expand ratio for inputs of MS-Block. Defaults to 3.
        in_down_ratio (float): Channel down ratio for downsample conv layer in MS-Block. Defaults to 1.
        mid_expand_ratio (float): Channel expand ratio for each branch in MS-Block. Defaults to 2.
        layers_num (int): Number of layer in MS-Block. Defaults to 3.
        kernel_sizes (list(int, tuple[int])): Sequential or number of kernel sizes in MS-Block. Defaults to [1,3,3].
    """
    def __init__(self,
                 in_expand_ratio: float = 1,
                 in_down_ratio: float = 1,
                 mid_expand_ratio: float = 2,
                 layers_num: int = 3,
                 kernel_sizes: Sequence[Union[int, Sequence[int]]] = [1,3,3],
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
            layers_num=int(self.layers_num * self.deepen_factor),
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
            layers_num=int(self.layers_num * self.deepen_factor),
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
