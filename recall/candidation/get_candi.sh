#!/bin/sh

root=$1
echo ${root}
raw_folder=$2
echo ${raw_folder}
out_folder=$3
echo ${out_folder}


# 生成item pair
nohup python -u get_candidation.py --input_file ${root}/data/sessions_train.csv \
  --output_file ${root}/cand/candi_1.csv \
  --product_file ${root}/data/products_train.csv \
  --root_path ${root} --topk 50 \
  --log_file ${root}/log/candi.log >> candi.log 2>&1 &


