#!/bin/sh

root=$1
echo ${root}
raw_folder=$2
echo ${raw_folder}
out_folder=$3
echo ${out_folder}


# 生成item pair
nohup python -u co_gen_item_pair.py --input_file ${root}/data/sessions_train.csv \
  --output_file ${root}/${out_folder}/pairs.${cnty}.csv \
  --country ${cnty} \
  --log_file ${root}/log/co_pair.log > co.log 2>&1 &

python -u co_calc_weight.py --input_file ${root}/${out_folder}/pairs.global.csv \
  --output_file ${root}/${out_folder}/co.global.txt \
  --log_file ${root}/log/co_pair.log

nohup python -u co_calc_weight.py --input_file ${root}/${out_folder}/pairs.global.csv \
  --output_file ${root}/${out_folder}/co.global.dict --topk 100 \
  --log_file ${root}/log/co_pair.log > co.log 2>&1 &

nohup python -u co_calc_weight.py --input_file ${root}/${out_folder}/pairs.DE.csv \
  --output_file ${root}/${out_folder}/co.DE.dict --topk 100 \
  --log_file ${root}/log/co_pair.log > co.log 2>&1 &

nohup python -u co_calc_weight.py --input_file ${root}/${out_folder}/pairs.JP.csv \
  --output_file ${root}/${out_folder}/co.JP.dict --topk 100 \
  --log_file ${root}/log/co_pair.log > co.log 2>&1 &

nohup python -u co_calc_weight.py --input_file ${root}/${out_folder}/pairs.UK.csv \
  --output_file ${root}/${out_folder}/co.UK.dict --topk 100 \
  --log_file ${root}/log/co_pair.log > co.log 2>&1 &

nohup python -u co_calc_weight.py --input_file ${root}/${out_folder}/pairs.ES.csv \
  --output_file ${root}/${out_folder}/co.ES.dict --topk 100 \
  --log_file ${root}/log/co_pair.log > co.log 2>&1 &
nohup python -u co_calc_weight.py --input_file ${root}/${out_folder}/pairs.FR.csv \
  --output_file ${root}/${out_folder}/co.FR.dict --topk 100 \
  --log_file ${root}/log/co_pair.log > co.log 2>&1 &
nohup python -u co_calc_weight.py --input_file ${root}/${out_folder}/pairs.IT.csv \
  --output_file ${root}/${out_folder}/co.IT.dict --topk 100 \
  --log_file ${root}/log/co_pair.log > co.log 2>&1 &