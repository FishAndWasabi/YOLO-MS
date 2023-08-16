# INSTALLATION

Installation is tested and working on the following platforms:

- Ubuntu 20.04.1
  - GPUs: RTX-3090 (Driver Version: 530.30.02)
  - RAM: 64GB
- CUDA==11.3
- pytorch==1.12.1
- torchvision=0.13.1


## Prerequisites

Our implementation based on MMYOLO==0.5.0. For more information about installation, please see the [official instructions](https://mmyolo.readthedocs.io/en/latest/).


**Step 0.** Create a virtual enviorment and activate it:

```bash
conda create -y -n YOLO-MS python=3.8
conda activate YOLO-MS
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
pip install -r requirements.txt
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

All the above instructions are integrated into install.sh.<br/>
So you can simply install by `bash install.sh`.