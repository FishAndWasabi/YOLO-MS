_base_ = '../yoloms/yoloms_rtmdet_m_syncbn_fast_8xb8-300e_coco.py'

model = dict(
    backbone=dict(
        out_attention_cfg=dict(type="SE")),
    neck=dict(
        out_attention_cfg=dict(type="SE"))
)