_base_ = '../yoloms/yoloms_rtmdet_tiny_syncbn_fast_8xb8-300e_coco.py'

layers_num=2
model = dict(
    backbone=dict(
        layers_num=layers_num),
    neck=dict(
        layers_num=layers_num)
)