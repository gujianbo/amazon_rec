#!/bin/sh

root=$1
echo ${root}

# 对计算的相似度进行过滤，保留每个topk
python -u recall_eval.py --input_file ${root}/data/sessions_train.csv --i2i_file ${root}/swing/swing.sim \
  --log_file ${root}/log/eval.log

root="/data/users/jianbogu/amazon/"
python -u recall_eval.py --input_file ${root}/data/sessions_train.csv --i2i_file ${root}/covisit/co.global.txt \
  --log_file ${root}/log/eval1.log