# pip3 install -U openmim
python3 -m mim install "mmengine>=0.6.0"
python3 -m mim install "mmmcv>=2.0.0rc4,<2.1.0"
python3 -m mim install "mmdet>=3.0.0rc6,<3.1.0"

# # pip3 install -r requirements/albu.txt
# # # Install MMYOLO
# # python3 -m mim install -v -e .

# # pip3 install tensorboard

# NAME=$1
# WORK_DIRS=$2
# bash tools/dist_train.sh ${NAME} 8 --resume --work-dir ${WORK_DIRS}