#!/bin/sh

root=$1
echo ${root}
raw_folder=$2
echo ${raw_folder}
out_folder=$3
echo ${out_folder}


# 生成item pair
nohup python -u get_candidation.py --input_file ${root}/data/sessions_train.csv \
  --output_file ${root}/cand/candi_100_top100.pre.csv --single_topk 100 \
  --product_file ${root}/data/products_train.csv \
  --root_path ${root} --topk 100 \
  --log_file ${root}/log/candi.log >> candi.log 2>&1 &

nohup python -u get_candidation.py --input_file ${root}/data/sessions_train.csv \
  --output_file ${root}/cand/candi_100_top200.lo.csv --single_topk 100 \
  --product_file ${root}/data/products_train.csv \
  --root_path ${root} --topk 200 \
  --log_file ${root}/log/candi.log >> candi.log 2>&1 &

nohup python -u get_candidation_sample.py --input_file ${root}/data/sessions_train.csv \
  --output_file ${root}/cand/candi_100_all.sample.lo.csv --single_topk 100 \
  --product_file ${root}/data/products_train.csv \
  --root_path ${root} --topk 200 --sample_cnt 20\
  --log_file ${root}/log/candi.log >> candi.log 2>&1 &

# 给提交数据预测
nohup python -u get_candidation.py --input_file ${root}/data/sessions_test_task1.csv \
  --output_file ${root}/cand/submission_100_top100.csv --single_topk 50 \
  --product_file ${root}/data/products_train.csv --is_train 0 \
  --root_path ${root} --topk 100 \
  --log_file ${root}/log/candi.log >> candi.log 2>&1 &

# 给提交数据预测
nohup python -u get_candidation.py --input_file ${root}/data/sessions_test_task1.csv \
  --output_file ${root}/cand/submission_100_top100.pre.csv --single_topk 100 \
  --product_file ${root}/data/products_train.csv --is_train 0 \
  --root_path ${root} --topk 100 \
  --log_file ${root}/log/candi.log >> candi.log 2>&1 &

nohup python -u flatten_candidation.py --input_file ${root}/cand/candi_100_top100.lo.csv \
  --output_file ${root}/cand/candi_100_top100.lo.flatten.csv \
  --log_file ${root}/log/flatten.log > flatten.log 2>&1 &

# 测试
nohup python -u flatten_candidation.py --input_file ${root}/cand/candi_100_top100.lo.csv \
  --output_file ${root}/cand/candi_100_top100.lo.test.flatten.csv \
  --log_file ${root}/log/flatten.log --neg_sample_rate 0 \
  --if_hash_sample 1 --sample_cnt 20 > flatten.log 2>&1 &



# 提交
nohup python -u flatten_candidation.py --input_file ${root}/cand/submission_100_top100.pre.csv \
  --output_file ${root}/cand/submission_100_top100.pre.flatten.csv \
  --log_file ${root}/log/flatten.log --neg_sample_rate 0 --drop_no_hit 0 > flatten.log 2>&1 &

