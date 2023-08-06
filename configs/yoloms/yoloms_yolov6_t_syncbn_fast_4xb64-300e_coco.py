_base_ = './yoloms_yolov6_t_syncbn_fast_8xb32-300e_coco.py'

train_batch_size_per_gpu = 64
train_dataloader = dict(batch_size=train_batch_size_per_gpu)