_base_ = '../yoloms/yoloms_rtmdet_tiny_syncbn_fast_8xb8-300e_coco.py'

layers_num=1
model = dict(
    neck=dict(
        _delete_=True,
        type='YOLOMSPAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[320, 640, 1280],
        feat_channels=[160, 320, 640],
        out_channels=out_channels,
        in_expand_ratio=3,
        mid_expand_ratio=2,
        layers_num = 3,
        kernel_sizes=[1,(3,3),(3,3)],
        in_down_ratio = 2,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True)
        ),
)