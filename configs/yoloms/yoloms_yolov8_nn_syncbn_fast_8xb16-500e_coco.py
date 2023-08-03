_base_ = './yoloms_yolov8_s_syncbn_fast_8xb16-500e_coco.py'

layers_num = 1
widen_factor = 0.15

model = dict(
    backbone=dict(layers_num=layers_num, 
                  widen_factor=widen_factor),
    neck=dict(layers_num=layers_num, 
              widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))
