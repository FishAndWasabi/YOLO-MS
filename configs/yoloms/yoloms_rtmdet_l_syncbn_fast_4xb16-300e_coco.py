_base_ = ['yoloms_rtmdet_l_syncbn_fast_8xb8-300e_coco.py']

train_batch_size_per_gpu = 16
train_dataloader = dict(batch_size=train_batch_size_per_gpu)