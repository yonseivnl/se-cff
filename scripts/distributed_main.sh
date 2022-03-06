#!/bin/bash

set -x

cuda_idx='6,7'
config_path=/root/code/configs/config.yaml
data_root=/root/data/DSEC
save_root=/root/code/save
num_workers=4
NUM_PROC=2

CUDA_VISIBLE_DEVICES=${cuda_idx} python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=$RANDOM ../src/distributed_main.py --config_path ${config_path} --data_root ${data_root} --save_root ${save_root} --num_workers ${num_workers}
