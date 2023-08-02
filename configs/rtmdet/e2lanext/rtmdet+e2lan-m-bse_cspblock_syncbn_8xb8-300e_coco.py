_base_ = './rtmdet+e2lan-l-bse_cspblock_syncbn_8xb8-300e_coco.py'

deepen_factor = 0.75
widen_factor = 0.7
out_channels = 240
model = dict(
    backbone=dict(
        arch='C3-K3579-80',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_expand_ratio=3,
        mid_expand_ratio=4,
        layers_num=3,
        down_ratio=1),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[320, 640, 1280],
        feat_channels=[240, 480, 960],
        out_channels=out_channels,
        num_csp_blocks=3),
    bbox_head=dict(
        head_module=dict(
            widen_factor=widen_factor,
            in_channels=out_channels,
            stacked_convs=2,
            feat_channels=out_channels)
        ),
)

visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), 
                                  dict(type='TensorboardVisBackend')])