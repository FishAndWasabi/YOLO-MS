#!/bin/bash

DEPLOY_CONFIG_FILE=$1
CONFIG_FILE=$2
CHECKPOINT_FILE=$3
IMAGE_FILE='mmyolo/demo/demo.jpg'
WORK_DIR=$4
DEVICE='cuda'

# generate engine file
python mmdeploy/tools/deploy.py $DEPLOY_CONFIG_FILE \
                                $CONFIG_FILE \
                                $CHECKPOINT_FILE \
                                $IMAGE_FILE \
                                --work-dir $WORK_DIR \
                                --device $DEVICE \
                                --dump-info
