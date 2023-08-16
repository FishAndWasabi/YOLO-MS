_base_ = 'mmyolo::yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py'

# ========================modified parameters======================
# Number of layer in MS-Block
layers_num = 3
# The scaling factor that controls the depth of the network structure
deepen_factor = 1/3
# The scaling factor that controls the width of the network structure
widen_factor = 0.15

# =======================Unmodified in most cases==================
# Channel expand ratio for inputs of MS-Block 
in_expand_ratio = 3
# Channel expand ratio for each branch in MS-Block 
mid_expand_ratio = 2
# Channel down ratio for downsample conv layer in MS-Block
in_down_ratio = 2

# The output channel of the last stage
last_stage_out_channels = 1280

# Kernel sizes of MS-Block in PAFPN
kernel_sizes = [1,(3,3),(3,3)]

# Normalization config
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001) 
# Activation config
act_cfg = dict(type='SiLU', inplace=True)

model = dict(
    backbone=dict(
        _delete_=True,
        type='YOLOv8MS',
        arch='C3-K3579',
        last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=norm_cfg,
        in_expand_ratio=in_expand_ratio,
        mid_expand_ratio=mid_expand_ratio,
        layers_num=layers_num,
        act_cfg=act_cfg),
    neck=dict(
        _delete_=True,
        type='YOLOv8MSPAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[320, 640, last_stage_out_channels],
        out_channels=[320, 640, last_stage_out_channels],
        in_expand_ratio=in_expand_ratio,
        in_down_ratio=in_down_ratio,
        mid_expand_ratio=mid_expand_ratio,
        kernel_sizes=kernel_sizes,
        layers_num=layers_num,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg),    
    bbox_head=dict(
        head_module=dict(
            widen_factor=widen_factor,
            in_channels=[320, 640, last_stage_out_channels])
        )
    )