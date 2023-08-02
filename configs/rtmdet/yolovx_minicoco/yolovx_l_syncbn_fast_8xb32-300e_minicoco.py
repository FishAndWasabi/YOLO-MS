_base_ = ['../yolovx/yolovx_l_syncbn_fast_8xb32-300e_coco.py']

data_root = '.'
# Path of train annotation file
train_ann_file = 'instances_minitrain2017.json'
train_data_prefix = 'data/coco/train2017/'  # Prefix of train image path

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix)
        )
)