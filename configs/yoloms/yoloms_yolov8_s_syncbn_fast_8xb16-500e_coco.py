_base_ = ['../yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py']

# The scaling factor that controls the depth of the network structure
deepen_factor = 1
layers_num = 1
# The scaling factor that controls the width of the network structure
widen_factor = 0.35
# The output channel of the last stage
last_stage_out_channels = 1280
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)  # Normalization config
train_batch_size_per_gpu=16

model = dict(
    backbone=dict(
        _delete_=True,
        type='YOLOMSv8',
        arch='C3-K3579-80',
        last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=norm_cfg,
        in_expand_ratio=3,
        mid_expand_ratio=2,
        layers_num=layers_num,
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        _delete_=True,
        type='YOLOMSv8PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[320, 640, last_stage_out_channels],
        out_channels=[320, 640, last_stage_out_channels],
        in_expand_ratio=3,
        in_down_ratio=2,
        mid_expand_ratio=2,
        kernel_sizes=[1,(3,3),(3,3)],
        layers_num=layers_num,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),    
    bbox_head=dict(
        head_module=dict(
            widen_factor=widen_factor,
            in_channels=[320, 640, last_stage_out_channels])))



train_dataloader = dict(batch_size=train_batch_size_per_gpu)
auto_scale_lr = dict(enable=True, base_batch_size=16*8)
