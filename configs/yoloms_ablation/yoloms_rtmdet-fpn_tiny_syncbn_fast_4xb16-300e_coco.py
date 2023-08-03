_base_ = '../yoloms/yoloms_rtmdet_tiny_syncbn_fast_4xb16-300e_coco.py'
widen_factor=0.4
model = dict(
    neck=dict(
        _delete_=True,
        type='mmdet.FPN',
        in_channels=[int(320*widen_factor), int(640*widen_factor), int(1280*widen_factor)],
        out_channels=int(240*widen_factor),
        num_outs=3,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True)
        ),
)