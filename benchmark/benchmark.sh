PATH=$1;
NAME=$2;

LOG="benchmark/${NAME}.log"
FILE="configs/${PATH}/${NAME}.py"
CHECKPOINT="yolovx_checkpoint/${NAME}.pth"

# echo ${LOG}
# echo ${FILE}
/home/ubuntu/anaconda3/envs/yolovx_deploy/bin/python tools/analysis_tools/get_flops.py ${FILE} | /usr/bin/tee ${LOG}
/usr/bin/sleep 30
int=1
while(( $int<=1 ))
do
    /home/ubuntu/anaconda3/envs/yolovx_deploy/bin/python tools/analysis_tools/benchmark.py ${FILE} --fuse-conv-bn --launcher none | /usr/bin/tee -a "${LOG}"
    let "int++"
    /usr/bin/sleep 10
done