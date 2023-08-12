## <p align=center> 💪 YOLO-MS: Rethinking Multi-Scale Representation Learning for Real-Time Object Detection </p>

![Python 3.8](https://img.shields.io/badge/python-3.8-g) ![pytorch 1.12.1](https://img.shields.io/badge/pytorch-1.12.0-blue.svg)

This repository contains the official implementation of the following paper:

> **YOLO-MS: Rethinking Multi-Scale Representation Learning for Real-Time Object Detection**<br/>
> [Yuming Chen](https://www.fishworld.site), [Xinbin Yuan](https://github.com/yuanxinbin), [Ruiqi Wu](https://scholar.google.com/citations?user=0ooNdgUAAAAJ&hl=en), [Jiabao Wang](https://scholar.google.co.uk/citations?hl=en&user=S9ErhhEAAAAJ), [Qibin Hou](https://houqb.github.io/), [Ming-ming Cheng](https://mmcheng.net)<br/>
> Under review

\[Homepage (TBD)\]
\[[Paper](https://arxiv.org/abs/2308.05480)]
\[Poster (TBD)\]
\[Video (TBD)\]



## Get Started

### 1. Prerequisites

- Ubuntu >= 20.04
- CUDA >= 11.3
- pytorch==1.12.1
- torchvision=0.13.1
- mmcv==2.0.0rc4
- mmengine==0.7.3

Our implementation based on MMDetection==3.0.0rc6. For more information about installation, please see the [official instructions](https://mmdetection.readthedocs.io/en/3.x/).


**Step 0.** Create Conda Environment

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab



### 2. Training

**Single GPU**

```shell
python tools/train.py configs/crosskd/${CONFIG_FILE} [optional arguments]
```

**Multi GPU**

```shell
CUDA_VISIBLE_DEVICES=x,x,x,x python tools/dist_train.sh \
    configs/crosskd/${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

### 3. Evaluation

```shell
python tools/test.py configs/crosskd/${CONFIG_FILE} ${CHECKPOINT_FILE}
```
