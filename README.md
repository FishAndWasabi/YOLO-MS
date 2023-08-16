## <p align=center> 🚀 YOLO-MS: Rethinking Multi-Scale Representation Learning for Real-Time Object Detection </p>

<div align="center">

![Python 3.8](https://img.shields.io/badge/python-3.8-g) 
![pytorch 1.12.1](https://img.shields.io/badge/pytorch-1.12.0-blue.svg)

</div>

This repository contains the official implementation of the following paper:

> **YOLO-MS: Rethinking Multi-Scale Representation Learning for Real-Time Object Detection**<br/>
> [Yuming Chen](https://www.fishworld.site), [Xinbin Yuan](https://github.com/yuanxinbin), [Ruiqi Wu](https://scholar.google.com/citations?user=0ooNdgUAAAAJ&hl=en), [Jiabao Wang](https://scholar.google.co.uk/citations?hl=en&user=S9ErhhEAAAAJ), [Qibin Hou](https://houqb.github.io/), [Ming-ming Cheng](https://mmcheng.net)<br/>
> Under review

\[Homepage (TBD)\]
\[[Paper](https://arxiv.org/abs/2308.05480)]
\[Poster (TBD)\]
\[Video (TBD)\]

<table>
  <tbody>
    <tr>
        <td>
            <img src='asserts/teaser_flops.png' alt='YOLOMS_TEASER0' width='500px'/>
        </td>
        <td>
            <img src='asserts/teaser_params.png' alt='YOLOMS_TEASER0' width='500px'/>
        </td>
    </tr>
    </tbody>
</table>


## 📄 Table of Contents

- [✨ New](#✨-news-)
- [🛠️ Dependencies and Installation](#🛠️-Dependencies-and-Installation-)
- [🤖 Training and Evaluation](#🤖-Training-and-Evaluation-)
- [🏡 Model Zoo](#🏡-Model-Zoo-)
- [🏗️ Other Task](#🏗️-Other-Task-)
- [📖 Citation](#📖-Citation-)
- [📜 License](#📜-License-)
- [📮 Contact](#📮-Contact-)
- [🤝 Acknowledgement](#🤝-Acknowledgement-)


## ✨ News

> Future work can be found in [todo.md](docs/todo.md).

- **Aug, 2023**: Our code is publicly available!


## 🛠️ Dependencies and Installation

We provide a simple scrpit `install.sh` for installation, or refer to [install.md](docs/install.md) for more details.

### 1. Clone and enter the repo.

```shell
git clone https://github.com/FishAndWasabi/YOLO-MS.git
cd YOLO-MS
```

### 2. Run `install.sh`.

```shell
bash install.sh
```

### 3. Activate your environment!

```shell
conda activate YOLO-MS
```


## 🤖 Training and Evaluation

### 1. Training

1.1 **Single GPU**

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

1.2 **Multi GPU**

```shell
CUDA_VISIBLE_DEVICES=x python tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

### 3. Evaluation

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
```

### 4. Deployment (TBD)



## 🏡 Model Zoo

### YOLOMS


### YOLOv6


### YOLOv8



## 📖 Citation

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

## 📜 License

Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only. Any commercial use should get formal permission first.

## 📮 Contact

For technical questions, please contact `chenyuming[AT]mail.nankai.edu.cn`.
For commercial licensing, please contact `cmm[AT]nankai.edu.cn` and `andrewhoux[AT]gmail.com`.

## 🤝 Acknowledgement

This repo is modified from open source real-time object detection codebase [MMYOLO](https://github.com/open-mmlab/mmyolo).
The README file is referred to [LED](https://github.com/Srameo/LED) and [CrossKD](https://github.com/jbwang1997/CrossKD)