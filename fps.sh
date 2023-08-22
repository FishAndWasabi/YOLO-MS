#!/bin/bash

DEPLOY_CONFIG_FILE=$1
CONFIG_FILE=$2
WORK_DIR=$3
DEVICE='cuda'
DATASET='data/coco/val2017'
PROFILER_MODEL="${WORK_DIR}/end2end.engine"

#test fps
total_latency=0
for i in {1..5}; do
    latency=$(python mmdeploy/tools/profiler.py ${DEPLOY_CONFIG_FILE} \
                                                ${CONFIG_FILE} \
                                                ${DATASET} \
                                                --model ${PROFILER_MODEL} \
                                                --device ${DEVICE} | awk 'NR==28{print $4}')
    echo "Latency $i in ms: $latency"
    total_latency=$(echo "$total_latency+$latency" | bc -l)
    echo "total_latency in ms: $total_latency"
done

avg_latency=$(echo "$total_latency / 5" | bc -l)
MODEL=`basename ${CONFIG_FILE}.py`
echo "$MODEL:" >> deploy_test.txt
printf "%.3f\n" "$avg_latency" >> deploy_test.txt
