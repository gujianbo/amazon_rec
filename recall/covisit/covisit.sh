#!/bin/sh

root=$1
echo ${root}
raw_folder=$2
echo ${raw_folder}
out_folder=$3
echo ${out_folder}


# 生成item pair
python -u co_gen_item_pair.py --input_file ${root}/${raw_folder}/sessions_train.csv \
  --output_file ${root}/${out_folder}/covisit.txt \
  --log_file ${root}/log/swing_pair.log