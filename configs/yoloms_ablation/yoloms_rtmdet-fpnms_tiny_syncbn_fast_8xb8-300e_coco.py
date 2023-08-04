_base_ = '../yoloms/yoloms_rtmdet_tiny_syncbn_fast_8xb8-300e_coco.py'
widen_factor=0.4
layers_num=1
model = dict(
    neck=dict(
        _delete_=True,
        type='YOLOMSFPN',
        in_channels=[320, 640, 1280],
        out_channels=240,
        num_outs=3,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True),
        in_expand_ratio=3,
        in_down_ratio=1,
        mid_expand_ratio=2,
        kernel_sizes=[1,3,3],
        layers_num=layers_num,
        widen_factor=widen_factor
        ),
)