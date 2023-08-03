_base_ = ['../yolov6/yolov6_s_syncbn_fast_8xb32-400e_coco.py']

# ======================= Possible modified parameters =======================
widen_factor = 0.35
layers_num = 1


# ============================== Unmodified in most cases ===================
model = dict(
    backbone=dict(
        _delete_=True,
        type='YOLOMSv6',
        arch='C3-K3579-80',
        widen_factor=widen_factor,
        in_expand_ratio=3,
        mid_expand_ratio=2,
        layers_num=layers_num,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        _delete_=True,
        type='YOLOMSv6PAFPN',
        widen_factor=widen_factor,
        in_channels=[320, 640, 1280],
        out_channels=[160, 320, 640],
        in_expand_ratio=3,
        in_down_ratio=2,
        mid_expand_ratio=2,
        kernel_sizes=[1,3,3],
        layers_num=layers_num,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    bbox_head=dict(
        head_module=dict(
            in_channels=[160, 320, 640],
            widen_factor=widen_factor)),
    )