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
  --country global \
  --log_file ${root}/log/co_pair.log

python -u co_calc_weight.py --input_file ${root}/${out_folder}/pairs.global.csv \
  --output_file ${root}/${out_folder}/co.global.txt \
  --log_file ${root}/log/co_pair.log

python -u co_calc_weight.py --input_file ${root}/${out_folder}/pairs.global.csv \
  --output_file ${root}/${out_folder}/co.global.txt \
  --log_file ${root}/log/co_pair.log