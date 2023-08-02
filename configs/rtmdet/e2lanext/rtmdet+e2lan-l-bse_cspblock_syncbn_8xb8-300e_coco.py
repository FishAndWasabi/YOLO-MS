_base_ = '../rtmdet_l_syncbn_fast_8xb32-300e_coco.py'

train_batch_size_per_gpu = 8
num_gpus=8
deepen_factor = 1
widen_factor = 1
out_channels = 260
model = dict(
    backbone=dict(
        _delete_=True,
        type='YOLONext',
        arch='C4-K3579-80',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_expand_ratio=1,
        mid_expand_ratio=4,
        layers_num=3,
        in_attention_cfg=None,
        mid_attention_cfg=None,
        out_attention_cfg=dict(type="SE"),
        down_ratio = 1,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[320, 640, 1280],
        feat_channels=[320, 640, 1280],
        out_channels=out_channels),
    bbox_head=dict(
        head_module=dict(
            widen_factor=widen_factor,
            in_channels=out_channels,
            stacked_convs=2,
            feat_channels=out_channels
            )
    )
)

train_dataloader = dict(batch_size=train_batch_size_per_gpu)
auto_scale_lr = dict(enable=True, base_batch_size=32*8)

visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), 
                                  dict(type='TensorboardVisBackend')])