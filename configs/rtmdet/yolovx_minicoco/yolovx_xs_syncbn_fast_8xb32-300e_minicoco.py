_base_ = ['yolovx_l_syncbn_fast_8xb32-300e_minicoco.py']
layers_num=2
deepen_factor = 1
widen_factor = 0.8
out_channels = 240
model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        layers_num=layers_num),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        layers_num=layers_num),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))