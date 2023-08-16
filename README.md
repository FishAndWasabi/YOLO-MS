<h2> <p align=center> 🚀 YOLO-MS: Rethinking Multi-Scale Representation Learning for Real-Time Object Detection </p> </h2>

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
\[知乎 (TBD)\]
\[[AIWalker](https://mp.weixin.qq.com/s/FfG9vNM_a2k_zflWfuimsw)\]
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


<h2> 📄 Table of Contents </h2>

- [✨ News 🔝](#-news-)
- [🛠️ Dependencies and Installation 🔝](#️-dependencies-and-installation-)
- [🤖 Training and Evaluation 🔝](#-training-and-evaluation-)
- [🏡 Model Zoo 🔝](#-model-zoo-)
- [🏗️ Supported Tasks 🔝](#️-supported-tasks-)
- [📖 Citation 🔝](#-citation-)
- [📜 License 🔝](#-license-)
- [📮 Contact 🔝](#-contact-)
- [🤝 Acknowledgement 🔝](#-acknowledgement-)


## ✨ News [🔝](#📄-table-of-contents)

> Future work can be found in [todo.md](docs/todo.md).

- **Aug, 2023**: Our code is publicly available!


## 🛠️ Dependencies and Installation [🔝](#📄-table-of-contents)

> We provide a simple scrpit `install.sh` for installation, or refer to [install.md](docs/install.md) for more details.

1. Clone and enter the repo.
    ```shell
    git clone https://github.com/FishAndWasabi/YOLO-MS.git
    cd YOLO-MS
    ```

2. Run `install.sh`.
    ```shell
    bash install.sh
    ```

3. Activate your environment!
    ```shell
    conda activate YOLO-MS
    ```


## 🤖 Training and Evaluation [🔝](#📄-table-of-contents)

1. Training

   1.1 Single GPU

   ```shell
   python tools/train.py ${CONFIG_FILE} [optional arguments]
   ```

   1.2 Multi GPU

   ```shell
   CUDA_VISIBLE_DEVICES=x python tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
   ```

2. Evaluation

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
```

3. Deployment (TBD)



## 🏡 Model Zoo [🔝](#📄-table-of-contents)

- [ ] YOLOv5-MS
- [ ] YOLOX-MS
- [x] [YOLOv6-MS](configs/yolov6)
- [ ] YOLOv7-MS
- [ ] PPYOLOE-MS
- [x] [YOLOv8-MS](configs/yolov8)
- [x] [YOLO-MS (Based on RTMDet)](configs/rtmdet)

<details>
<summary><b>1. YOLO-MS</b></summary>

<table>
    <thead>
    <tr>
        <th style="width: 300pt"> Model </th>
        <th> Resolution </th>
        <th> Epoch </th>
        <th> Params(M) </th>
        <th> FLOPs(G) </th>
        <th> boxAP </th>
        <th> boxAP(small) </th>
        <th> boxAP(middle) </th>
        <th> boxAP(large) </th>
        <th> Config File </th>
        <th> 🔗 Download Links </th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td style="width: 300pt"> YOLO-MS-XS </td>
        <td> 640 </td>
        <td> 300 </td>
        <td> Params(M) </td>
        <td> FLOPs(G) </td>
        <td> boxAP </td>
        <td> boxAP(small) </td>
        <td> boxAP(small) </td>
        <td> boxAP(large) </td>
        <td> [<a href="./configs/yoloms/yoloms-xs_syncbn_fast_8xb8-300e_coco.py">config</a>]  </td>
        <td> [<a href="">model</a>] </td>
    </tr>
    <tr>
        <td> YOLO-MS-S </td>
        <td> 640 </td>
        <td> 300 </td>
        <td> Params(M) </td>
        <td> FLOPs(G) </td>
        <td> boxAP </td>
        <td> boxAP(small) </td>
        <td> boxAP(small) </td>
        <td> boxAP(large) </td>
        <td> [<a href="./configs/yoloms/yoloms-xs_syncbn_fast_8xb8-300e_coco.py">config</a>]  </td>
        <td> [<a href="">model</a>] </td>
    </tr>
    <tr>
        <td> YOLO-MS </td>
        <td> 640 </td>
        <td> 300 </td>
        <td> Params(M) </td>
        <td> FLOPs(G) </td>
        <td> boxAP </td>
        <td> boxAP(small) </td>
        <td> boxAP(small) </td>
        <td> boxAP(large) </td>
        <td> [<a href="./configs/yoloms/yoloms-xs_syncbn_fast_8xb8-300e_coco.py">config</a>]  </td>
        <td> [<a href="">model</a>] </td>
    </tr>
    </tbody>
</table>

</details>

<details>
<summary><b>2. YOLOv6-MS</b></summary>


</details>

<details>
<summary><b>3. YOLOv8-MS</b></summary>


</details>


## 🏗️ Supported Tasks [🔝](#📄-table-of-contents)

- [x] Object Detection
- [ ] Instance Segmentation (TBD)
- [ ] Rotated Object Detection (TBD)
- [ ] Object Tracking (TBD)
- [ ] Detection in Crowded Scene (TBD)
- [ ] Small Object Detection (TBD)


## 📖 Citation [🔝](#📄-table-of-contents)

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

This project is based on the open source codebase [MMYOLO](https://github.com/open-mmlab/mmyolo).

```
@misc{mmyolo2022,
    title={{MMYOLO: OpenMMLab YOLO} series toolbox and benchmark},
    author={MMYOLO Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmyolo}},
    year={2022}
}
```


## 📜 License [🔝](#📄-table-of-contents)

Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only. Any commercial use should get formal permission first.


## 📮 Contact [🔝](#📄-table-of-contents)

For technical questions, please contact `chenyuming[AT]mail.nankai.edu.cn`.
For commercial licensing, please contact `cmm[AT]nankai.edu.cn` and `andrewhoux[AT]gmail.com`.


## 🤝 Acknowledgement [🔝](#📄-table-of-contents)

This repo is modified from open source real-time object detection codebase [MMYOLO](https://github.com/open-mmlab/mmyolo).
The README file is referred to [LED](https://github.com/Srameo/LED) and [CrossKD](https://github.com/jbwang1997/CrossKD)