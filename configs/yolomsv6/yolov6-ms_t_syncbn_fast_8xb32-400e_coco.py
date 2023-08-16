_base_ = 'mmyolo::yolov6/yolov6_s_syncbn_fast_8xb32-400e_coco.py'

# ========================modified parameters======================
# Number of layer in MS-Block
layers_num = 3
# The scaling factor that controls the depth of the network structure
deepen_factor = 1/3
# The scaling factor that controls the width of the network structure
widen_factor = 0.35

# Input channels of PAFPN
in_channels=[320, 640, 1280]
# Output channels of PAFPN
out_channels=[160, 320, 640]

# ============================== Unmodified in most cases ===================
# Channel expand ratio for inputs of MS-Block 
in_expand_ratio = 3
# Channel expand ratio for each branch in MS-Block 
mid_expand_ratio = 2
# Channel down ratio for downsample conv layer in MS-Block
in_down_ratio = 2

# Kernel sizes of MS-Block in PAFPN
kernel_sizes = [1,(3,3),(3,3)]

# Normalization config
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001) 
# Activation config
act_cfg = dict(type='SiLU', inplace=True)

model = dict(
    backbone=dict(
        _delete_=True,
        type='YOLOv6MS',
        arch='C3-K3579',
        widen_factor=widen_factor,
        in_expand_ratio=in_expand_ratio,
        mid_expand_ratio=mid_expand_ratio,
        layers_num=layers_num,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg),
    neck=dict(
        _delete_=True,
        type='YOLOv6MSPAFPN',
        widen_factor=widen_factor,
        in_channels=in_channels,
        out_channels=out_channels,
        in_expand_ratio=in_expand_ratio,
        in_down_ratio=in_down_ratio,
        mid_expand_ratio=mid_expand_ratio,
        kernel_sizes=kernel_sizes,
        layers_num=layers_num,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg),
    bbox_head=dict(
        head_module=dict(
            in_channels=out_channels,
            widen_factor=widen_factor)
        )
    )
