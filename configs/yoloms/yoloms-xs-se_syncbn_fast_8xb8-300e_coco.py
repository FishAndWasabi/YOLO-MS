_base_ = './yoloms-s_syncbn_fast_8xb8-300e_coco.py'

# ========================modified parameters======================
# The scaling factor that controls the depth of the network structure
deepen_factor = 1/3
# The scaling factor that controls the width of the network structure
widen_factor =  0.4

model = dict(backbone=dict(deepen_factor=deepen_factor,
                           widen_factor=widen_factor,
                           attention_cfg=dict(type="SE")),
             neck=dict(deepen_factor=deepen_factor,
                       widen_factor=widen_factor),
             bbox_head=dict(head_module=dict(widen_factor=widen_factor)))