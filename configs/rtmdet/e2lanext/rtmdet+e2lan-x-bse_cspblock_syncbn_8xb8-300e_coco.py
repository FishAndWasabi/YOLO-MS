_base_ = './rtmdet+e2lan-l-bse_cspblock_syncbn_8xb8-300e_coco.py'

deepen_factor = 1.33
widen_factor = 1.2
out_channels = 250
model = dict(
    backbone=dict(
        arch='C3333-K3579-80',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        downsample_ratio=1,
        first_expand_ratio=4),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[320, 640, 1280],
        feat_channels=[320, 640, 1280],
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