FILE="yoloms/yoloms_yolov8_nn_syncbn_fast_8xb16-500e_coco"
python tools/analysis_tools/benchmark.py configs/${FILE}.py | tee log/${FILE}.log

FILE="yoloms/yoloms_yolov8_n_syncbn_fast_8xb16-500e_coco"
python tools/analysis_tools/benchmark.py configs/${FILE}.py | tee log/${FILE}.log

FILE="yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco"
python tools/analysis_tools/benchmark.py configs/${FILE}.py | tee log/${FILE}.log