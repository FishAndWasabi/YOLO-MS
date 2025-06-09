# Reference to
# https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py
_base_ = '/mnt/data1/workspace/wmq/YOLO-World/third_party/mmyolo/configs/yolov8/yolov8_l_syncbn_fast_8xb16-500e_coco.py'

# ========================Frequently modified parameters======================
# -----data related-----
data_root = '/mnt/data1/workspace/wmq/YOLO-World/data/qiyuan2/'  # Root path of data
# Path of train annotation file
train_ann_file = 'instances_trainval.json'
train_data_prefix = 'trainval/images'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'instances_val.json'
val_data_prefix = 'val/images'  # Prefix of val image path

num_classes = 10  # Number of classes for classification

# -----model related-----
# Number of layer in MS-Block
layers_num = 3
# The scaling factor that controls the depth of the network structure
deepen_factor = 1.0
# The scaling factor that controls the width of the network structure
widen_factor = 1.0

# Channel expand ratio for inputs of MS-Block
in_expand_ratio = 3
# Channel expand ratio for each branch in MS-Block
mid_expand_ratio = 2
# Channel down ratio for downsample conv layer in MS-Block
in_down_ratio = 2

# The output channel of the last stage
last_stage_out_channels = 512

# Kernel sizes of MS-Block in PAFPN
kernel_sizes = [1, (3, 3), (3, 3)]

# Normalization config
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
# Activation config
act_cfg = dict(type='SiLU', inplace=True)

# =======================Unmodified in most cases==================
model = dict(
    backbone=dict(_delete_=True,
                  type='YOLOv8MS',
                  arch='C4-K3579',
                  last_stage_out_channels=last_stage_out_channels,
                  deepen_factor=deepen_factor,
                  widen_factor=widen_factor,
                  norm_cfg=norm_cfg,
                  in_expand_ratio=in_expand_ratio,
                  mid_expand_ratio=mid_expand_ratio,
                  layers_num=layers_num,
                  act_cfg=act_cfg),
    neck=dict(_delete_=True,
              type='YOLOv8MSPAFPN',
              deepen_factor=deepen_factor,
              widen_factor=widen_factor,
              in_channels=[256, 512, last_stage_out_channels],
              out_channels=[256, 512, last_stage_out_channels],
              in_expand_ratio=in_expand_ratio,
              in_down_ratio=in_down_ratio,
              mid_expand_ratio=mid_expand_ratio,
              kernel_sizes=kernel_sizes,
              layers_num=layers_num,
              norm_cfg=norm_cfg,
              act_cfg=act_cfg),
    bbox_head=dict(
        head_module=dict(widen_factor=widen_factor,
                         num_classes=num_classes,
                         in_channels=[256, 512, last_stage_out_channels])),
    train_cfg=dict(assigner=dict(num_classes=num_classes)),
)

train_dataloader = dict(pin_memory=True,
                        sampler=dict(_delete_=True,
                                     type='DefaultSampler',
                                     shuffle=True),
                        collate_fn=dict(_delete_=True, type='yolov5_collate'),
                        dataset=dict(data_root=data_root,
                                     ann_file=train_ann_file,
                                     filter_cfg=dict(filter_empty_gt=True),
                                     metainfo={'classes':
                                         ('person', 'car', 'ship', 'plane', 'truck', 'van', 'bus', 'motor', 'bicycle', 'tricycle')},
                                     data_prefix=dict(img=train_data_prefix)))

val_dataloader = dict(pin_memory=True,
                      drop_last=False,
                      sampler=dict(_delete_=True,
                                   type='DefaultSampler',
                                   shuffle=False),
                      dataset=dict(data_root=data_root,
                                   test_mode=True,
                                   data_prefix=dict(img=val_data_prefix),
                                   ann_file=val_ann_file,
                                   metainfo={'classes':
                                       ('person', 'car', 'ship', 'plane', 'truck', 'van', 'bus', 'motor', 'bicycle', 'tricycle')},))

test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + val_ann_file,)