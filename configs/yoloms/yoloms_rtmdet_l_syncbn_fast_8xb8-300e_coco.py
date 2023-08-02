_base_ = ['yoloms_rtmdet_l_syncbn_fast_8xb32-300e_coco.py']

train_batch_size_per_gpu = 8
train_dataloader = dict(batch_size=train_batch_size_per_gpu)
auto_scale_lr = dict(enable=True, base_batch_size=32*8)