#!/bin/sh

root=$1
echo ${root}

nohup python -u dnn_train.py --train_file ${root}/dnn_train/ \
  --test_file ${root}/dnn_test/test.123 --max_seq_len 40 \
  --dense_size 89 --train_batch_size 64 --test_batch_size 32 \
  --save_path ${root}/models/ --log_file ${root}/log/dnn.log > log 2>&1 &