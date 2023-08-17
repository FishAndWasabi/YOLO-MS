#!/bin/bash

CONFIG_FILE=$1  
CONFIG_FILE2=$2
CHECKPOINT_FILE=$3
IMAGE_FILE='mmyolo/demo/demo.jpg'
WORK_DIR=$4
DEVICE='cuda'
DATASET='data/coco/val2017'
PROFILER_MODEL="$WORK_DIR/end2end.engine"

# generate engine file 
python mmdeploy/tools/deploy1.py $CONFIG_FILE $CONFIG_FILE2 $CHECKPOINT_FILE $IMAGE_FILE --work-dir $WORK_DIR --device $DEVICE --dump-info

#test fps
total_latency=0
for i in {1..5}; do
    latency=$(python mmdeploy/tools/profiler.py $CONFIG_FILE $CONFIG_FILE2 $DATASET --model $PROFILER_MODEL --device $DEVICE | awk 'NR==28{print $4}')
    echo "Latency $i in ms: $latency"
    total_latency=$(echo "$total_latency+$latency" | bc -l)
    echo "total_latency in ms: $total_latency"
done

avg_latency=$(echo "$total_latency / 5" | bc -l)
MODEL=`basename ${CONFIG_FILE2} .py`
echo "$MODEL:" >> deploy_test.txt
printf "%.3f\n" "$avg_latency" >> deploy_test.txt