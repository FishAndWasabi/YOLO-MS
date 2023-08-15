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

Our implementation based on MMYOLO==0.5.0. For more information about installation, please see the [official instructions](https://mmyolo.readthedocs.io/en/latest/).


**Step 0.** Create Conda Environment

```shell
conda create --name yolo-ms python=3.8 -y
conda activate yolo-ms
```

**Step 1.** Install [Pytorch](https://pytorch.org)

```shell
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

**Step 2.** Install [MMEngine](https://github.com/open-mmlab/mmengine), [MMCV](https://github.com/open-mmlab/mmcv), [MMDet](https://github.com/open-mmlab/mmdet) and [MMYOLO](https://github.com/open-mmlab/mmyolo) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install "mmengine==0.7.1"
mim install "mmcv==2.0.0rc4"
mim install "mmdet==3.0.0"
mim install "mmyolo==0.5.0"
```

**Step 3.** Install YOLO-MS.

```shell
git clone https://github.com/FishAndWasabi/YOLO-MS.git
cd YOLO-MS
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

### 2. Training

**Single GPU**

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

**Multi GPU**

```shell
CUDA_VISIBLE_DEVICES=x,x,x,x python tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

### 3. Evaluation

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
```


### 4. Deployment

TODO





## Results


