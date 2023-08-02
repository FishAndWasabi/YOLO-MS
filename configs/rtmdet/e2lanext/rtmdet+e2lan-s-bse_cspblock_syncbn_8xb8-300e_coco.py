_base_ = './rtmdet+e2lan-l-bse_cspblock_syncbn_8xb8-300e_coco.py'

checkpoint = 'work_dirs/xxx.pth'  # noqa
train_batch_size_per_gpu = 32
num_gpus=8
deepen_factor = 0.33
widen_factor = 0.5
out_channels = 256
img_scale = _base_.img_scale

model = dict(
    backbone=dict(
        arch='C222-K3579-80',
        expand_ratio=0.5,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        downsample_ratio=1,
        first_expand_ratio=3),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[320, 640, 1280],
        feat_channels=[160, 320, 640],
        out_channels=out_channels,
        num_csp_blocks=3,
        expand_ratio=0.5),
    bbox_head=dict(
        head_module=dict(
            num_classes=80,
            widen_factor=widen_factor,
            in_channels=out_channels,
            stacked_convs=2,
            feat_channels=out_channels),
        ),
)


train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Mosaic',
        img_scale=img_scale,
        use_cached=True,
        max_cached_images=40,
        pad_val=114.0),
    dict(
        type='mmdet.RandomResize',
        scale=(img_scale[0] * 2, img_scale[1] * 2),
        ratio_range=(0.5, 2.0),  # note
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='YOLOXMixUp',
        img_scale=img_scale,
        use_cached=True,
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=(114, 114, 114)),
    dict(type='mmdet.PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.RandomResize',
        scale=img_scale,
        ratio_range=(0.5, 2.0),  # note
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=280,
        switch_pipeline=train_pipeline_stage2)
]

visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), 
                                  dict(type='TensorboardVisBackend')])