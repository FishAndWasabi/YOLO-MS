_base_ = '../yoloms/yoloms_rtmdet_tiny_syncbn_fast_8xb8-300e_coco.py'
widen_factor=0.45
layers_num=1
out_channels=256

model = dict(
    backbone=dict(
        out_attention_cfg=None,
        widen_factor=widen_factor,
        layers_num=layers_num),
    neck=dict(
        _delete_=True,
        type='YOLOMSNeck',
        in_channels=[320, 640, 1280],
        feat_channels=[128, 256, 512],
        out_channels=out_channels,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True),
        in_expand_ratio=3,
        in_down_ratio=1,
        mid_expand_ratio=2,
        kernel_sizes=[1,3,3],
        layers_num=layers_num,
        widen_factor=widen_factor
        ),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor,
                                    in_channels=out_channels,
                                    stacked_convs=2,
                                    feat_channels=out_channels))
    )