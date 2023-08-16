# setting conda hook
eval "$(conda shell.bash hook)"

# create env
conda create -y -n YOLO-MS python=3.8
conda activate YOLO-MS

# install pytorch
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# install dependencies of openmmlab
pip install -U openmim
mim install "mmengine==0.7.1"
mim install "mmcv==2.0.0rc4"
mim install "mmdet==3.0.0"
mim install "mmyolo==0.5.0"

# install other dependencies
pip install -r requirements.txt

# install yolo-ms
pip install -v -e .