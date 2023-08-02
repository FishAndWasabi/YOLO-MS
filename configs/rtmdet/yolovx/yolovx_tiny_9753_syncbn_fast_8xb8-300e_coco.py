_base_ = './yolovx_s_syncbn_fast_8xb8-300e_coco.py'

# ========================modified parameters======================
layers_num = 1
deepen_factor = 1
widen_factor = 0.4
out_channels=240

img_scale = _base_.img_scale

model = dict(
    backbone=dict(
        arch='C3-K9753-80',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        layers_num=layers_num),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        layers_num=layers_num),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))


visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), 
                                  dict(type='WandbVisBackend',
                                       init_kwargs=dict(project="YoloVX",
                                                        group="coco_8", 
                                                        name="yolovx_tiny_9999_syncbn_fast_8xb32-300e_coco",
                                                        resume="auto"))])