#!/bin/sh

root=$1
echo ${root}

# 对计算的相似度进行过滤，保留每个topk
python -u swing_sim_uniq.py --input_file ${root}/data/sessions_train.csv --i2i_file ${root}/swing/swing.sim \
  --log_file ${root}/log/eval.log