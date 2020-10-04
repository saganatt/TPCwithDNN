#!/bin/bash

# 6 GPUs
# sed -i 's/run_parallel:\ false/run_parallel:\ true/' database_parameters_DNN_fluctuations.yml
# sed -i "s/\"\/gpu:0\",\ \"\/gpu:1\"/\"\/gpu:0\",\ \"\/gpu:1\",\ \"\/gpu:2\",\ \"\/gpu:3\",\ \"\/gpu:4\",\ \"\/gpu:5\"/" dnn_optimiser.py
# echo "Running for 6 GPUs"
# for train in 1000 2000 4000; do
#     test=$((train / 10))
#     sed -i "s/train_events:\ \[[1-9]000\]/train_events:\ \[${train}\]/" database_parameters_DNN_fluctuations.yml
#     sed -i "s/test_events:\ \[[1-9]00\]/test_events:\ \[${test}\]/" database_parameters_DNN_fluctuations.yml
#     echo "${train} train events, ${test} test events"
#     # /opt/rocm/bin/rocprof --sys-trace python steer_analysis.py > debug.txt 2>&1
#     time python steer_analysis.py > debug.txt 2>&1
#     DIRNAME=run_6-train_${train}
#     mkdir -p $DIRNAME 
#     mv plots/ $DIRNAME/
#     mv model_new_random/ $DIRNAME/
#     mv validation_new_random/ $DIRNAME/
#     mv 2020* $DIRNAME/
#     mv debug.txt $DIRNAME/
# done
# 
# # 4 GPUs
# sed -i "s/,\ \"\/gpu:4\",\ \"\/gpu:5\"//" dnn_optimiser.py
# echo "Running for 4 GPUs"
# for train in 1000 2000 4000; do
#     test=$((train / 10))
#     sed -i "s/train_events:\ \[[1-9]000\]/train_events:\ \[${train}\]/" database_parameters_DNN_fluctuations.yml
#     sed -i "s/test_events:\ \[[1-9]00\]/test_events:\ \[${test}\]/" database_parameters_DNN_fluctuations.yml
#     echo "${train} train events, ${test} test events"
#     # /opt/rocm/bin/rocprof --sys-trace python steer_analysis.py > debug.txt 2>&1
#     time python steer_analysis.py > debug.txt 2>&1
#     DIRNAME=run_4-train_${train}
#     mkdir -p $DIRNAME 
#     mv plots/ $DIRNAME/
#     mv model_new_random/ $DIRNAME/
#     mv validation_new_random/ $DIRNAME/
#     mv 2020* $DIRNAME/
#     mv debug.txt $DIRNAME/
# done

# 2 GPUs
# sed -i "s/,\ \"\/gpu:2\",\ \"\/gpu:3\"//" dnn_optimiser.py
# echo "Running for 2 GPUs"
# for train in 1000 2000 4000; do
#     test=$((train / 10))
#     sed -i "s/train_events:\ \[[1-9]000\]/train_events:\ \[${train}\]/" database_parameters_DNN_fluctuations.yml
#     sed -i "s/test_events:\ \[[1-9]00\]/test_events:\ \[${test}\]/" database_parameters_DNN_fluctuations.yml
#     echo "${train} train events, ${test} test events"
#     # /opt/rocm/bin/rocprof --sys-trace python steer_analysis.py > debug.txt 2>&1
#     time python steer_analysis.py > debug.txt 2>&1
#     DIRNAME=run_2-train_${train}
#     mkdir -p $DIRNAME 
#     mv plots/ $DIRNAME/
#     mv model_new_random/ $DIRNAME/
#     mv validation_new_random/ $DIRNAME/
#     mv 2020* $DIRNAME/
#     mv debug.txt $DIRNAME/
# done

# 1 GPU
sed -i 's/run_parallel:\ true/run_parallel:\ false/' database_parameters_DNN_fluctuations.yml
echo "Running for 1 GPU"
for train in 1000 2000 4000; do
    test=$((train / 10))
    sed -i "s/train_events:\ \[[1-9]000\]/train_events:\ \[${train}\]/" database_parameters_DNN_fluctuations.yml
    sed -i "s/test_events:\ \[[1-9]00\]/test_events:\ \[${test}\]/" database_parameters_DNN_fluctuations.yml
    echo "${train} train events, ${test} test events"
    # /opt/rocm/bin/rocprof --sys-trace python steer_analysis.py > debug.txt 2>&1
    time python steer_analysis.py > debug.txt 2>&1
    DIRNAME=run_1-train_${train}
    mkdir -p $DIRNAME 
    mv plots/ $DIRNAME/
    mv model_new_random/ $DIRNAME/
    mv validation_new_random/ $DIRNAME/
    mv 2020* $DIRNAME/
    mv debug.txt $DIRNAME/
done
