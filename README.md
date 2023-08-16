## <p align=center> 💪 YOLO-MS: Rethinking Multi-Scale Representation Learning for Real-Time Object Detection </p>

<!-- <center>
![Python 3.8](https://img.shields.io/badge/python-3.8-g) ![pytorch 1.12.1](https://img.shields.io/badge/pytorch-1.12.0-blue.svg)
<center/> -->

This repository contains the official implementation of the following paper:

> **YOLO-MS: Rethinking Multi-Scale Representation Learning for Real-Time Object Detection**<br/>
> [Yuming Chen](https://www.fishworld.site), [Xinbin Yuan](https://github.com/yuanxinbin), [Ruiqi Wu](https://scholar.google.com/citations?user=0ooNdgUAAAAJ&hl=en), [Jiabao Wang](https://scholar.google.co.uk/citations?hl=en&user=S9ErhhEAAAAJ), [Qibin Hou](https://houqb.github.io/), [Ming-ming Cheng](https://mmcheng.net)<br/>
> Under review

\[Homepage (TBD)\]
\[[Paper](https://arxiv.org/abs/2308.05480)]
\[Poster (TBD)\]
\[Video (TBD)\]




<img src='assets/teaser_params.png' alt='YOLOMS_TEASER0' width='500px'/>


- First of all, [:wrench: Dependencies and Installation](#wrench-dependencies-and-installation).
- For **academic research**, please refer to [pretrained-models.md](docs/pretrained-models.md) and [:robot: Training and Evaluation](#robot-training-and-evaluation).
- For **further development**, please refer to [:construction: Further Development](#construction-further-development).
- For **using LED on your own camera**, please refer to [:sparkles: Pretrained Models](#sparkles-pretrained-models) and [:camera: Quick Demo](#camera-quick-demo).




## Get Started

### 1. Dependencies and Installation

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

### 2. Training and Evaluation

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





## :sparkles: Model Zoo

### YOLOMS


### YOLOv6


### YOLOv8



## :book: Citation

If you find our repo useful for your research, please cite us:

```
@misc{chen2023yoloms,
      title={YOLO-MS: Rethinking Multi-Scale Representation Learning for Real-time Object Detection}, 
      author={Yuming Chen and Xinbin Yuan and Ruiqi Wu and Jiabao Wang and Qibin Hou and Ming-Ming Cheng},
      year={2023},
      eprint={2308.05480},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

This project is based on the open source codebase [MMDetection](https://github.com/open-mmlab/mmdetection).
```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```

## :scroll: License

Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only. Any commercial use should get formal permission first.

## :postbox: Contact

For technical questions, please contact `chenyuming@mail.nankai.edu.cn` and ``.

## :handshake: Acknowledgement

This repo is modified from open source object detection codebase [MMDetection](https://github.com/open-mmlab/mmdetection).


We also thank all of our contributors.

<a href="https://github.com/Srameo/LED/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Srameo/LED" />
</a>