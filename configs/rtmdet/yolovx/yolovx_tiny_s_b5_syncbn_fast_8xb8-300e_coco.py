_base_ = './yolovx_s_syncbn_fast_8xb8-300e_coco.py'

# ========================modified parameters======================
layers_num = 1
deepen_factor = 1
widen_factor = 0.35
out_channels=240

img_scale = _base_.img_scale

model = dict(
    backbone=dict(
        arch='C5-K3579-80',
        in_expand_ratio=5,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        layers_num=layers_num),
    neck=dict(
        deepen_factor=deepen_factor,
        in_expand_ratio=5,
        widen_factor=widen_factor,
        kernel_sizes=[1,(3,3),(3,3),(3,3),(3,3)],
        layers_num=layers_num),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))