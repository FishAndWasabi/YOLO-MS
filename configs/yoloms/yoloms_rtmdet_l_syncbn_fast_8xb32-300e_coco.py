_base_ = ['../rtmdet/rtmdet_l_syncbn_fast_8xb32-300e_coco.py']

loss_bbox_weight = 2.0
train_batch_size_per_gpu = 32
num_gpus=8
deepen_factor = 1
widen_factor = 1
out_channels = 240
model = dict(
    backbone=dict(
        _delete_=True,
        type='YOLOMS',
        arch='C3-K3579-80',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_expand_ratio=3,
        mid_expand_ratio=2,
        layers_num=3,
        in_attention_cfg=None,
        mid_attention_cfg=None,
        out_attention_cfg=dict(type="SE"),
        down_ratio = 1,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True)),
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
        in_attention_cfg=None,
        mid_attention_cfg=None,
        out_attention_cfg=None,
        kernel_sizes=[1,(3,3),(3,3)],
        in_down_ratio = 2,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True)
        ),
    bbox_head=dict(
        type='RTMDetHead',
        head_module=dict(
            type='RTMDetSepBNHeadModule',
            num_classes=80,
            widen_factor=widen_factor,
            in_channels=out_channels,
            stacked_convs=2,
            feat_channels=out_channels,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='LeakyReLU', inplace=True),
            share_conv=True,
            pred_kernel_size=1,
            featmap_strides={{_base_.strides}}),
        loss_bbox=dict(type='mmdet.DIoULoss', loss_weight=loss_bbox_weight))
)

train_dataloader = dict(batch_size=train_batch_size_per_gpu)
auto_scale_lr = dict(enable=True, base_batch_size=32*8)

visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), 
                                  dict(type='TensorboardVisBackend')])