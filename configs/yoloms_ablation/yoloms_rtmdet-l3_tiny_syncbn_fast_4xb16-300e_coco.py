_base_ = '../yoloms/yoloms_rtmdet_tiny_syncbn_fast_4xb16-300e_coco.py'

layers_num=3
model = dict(
    backbone=dict(
        layers_num=layers_num),
    neck=dict(
        layers_num=layers_num)
)