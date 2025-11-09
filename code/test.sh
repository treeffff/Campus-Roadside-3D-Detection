#!/bin/bash
# train.sh

# 切换到 project 根目录
cd "$(dirname "$0")/.."   # code/ 的上一级就是 project/

# 运行训练脚本
python ./code/exps/bev_height_lss_r101_864_1536_256x256_140_test.py \
    --ckpt_path ./user_data/model_weights \
    -e -b 4 --gpus 1
