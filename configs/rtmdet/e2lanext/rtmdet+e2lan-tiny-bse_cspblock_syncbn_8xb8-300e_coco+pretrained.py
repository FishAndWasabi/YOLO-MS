_base_ = './rtmdet+e2lan-s-bse_cspblock_syncbn_8xb8-300e_coco.py'

# checkpoint = 'work_dirs/e2lan-tiny0.4-1+3+C222-K3579-80-bse_b256-rsb-a1-300e_in1k+syncbn+adamw/epoch_300.pth'  # noqa
num_gpus=8
deepen_factor = 0.167
widen_factor = 0.4
img_scale = _base_.img_scale
model = dict(
    backbone=dict(
        arch='C222-K3579-80',
        expand_ratio=0.5,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        downsample_ratio=1,
        first_expand_ratio=3,
        # init_cfg=dict(
        #     type='Pretrained',
        #     prefix='backbone.',
        #     checkpoint=checkpoint,
        #     map_location='cpu')
        ),
    neck=dict(
        type='CSPNeXtPAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[320, 640, 1280],
        feat_channels=[160, 320, 640],
        out_channels=240,
        num_csp_blocks=3,
        expand_ratio=0.5),
    bbox_head=dict(
        head_module=dict(
            widen_factor=widen_factor,
            in_channels=240,
            stacked_convs=2,
            feat_channels=240),
    )
)


train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Mosaic',
        img_scale=img_scale,
        use_cached=True,
        max_cached_images=20,  # note
        random_pop=False,  # note
        pad_val=114.0),
    dict(
        type='mmdet.RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.5, 2.0),
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='YOLOXMixUp',
        img_scale=img_scale,
        ratio_range=(1.0, 1.0),
        max_cached_images=10,  # note
        use_cached=True,
        random_pop=False,  # note
        pad_val=(114, 114, 114),
        prob=0.5),  # note
    dict(type='mmdet.PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
