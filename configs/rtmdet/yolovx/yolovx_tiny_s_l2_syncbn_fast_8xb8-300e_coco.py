_base_ = './yolovx_s_syncbn_fast_8xb8-300e_coco.py'

# ========================modified parameters======================
layers_num = 1
deepen_factor = 1
widen_factor = 0.4
out_channels=240

img_scale = _base_.img_scale

model = dict(
    backbone=dict(
        arch='C3-K3579-80-s',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        layers_num=2),
    neck=dict(
        layer_type = "e2lan",
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        layers_num=2),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))

visualizer = dict(vis_backends = [dict(type='LocalVisBackend')])