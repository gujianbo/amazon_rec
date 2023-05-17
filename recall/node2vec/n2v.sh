#!/bin/sh

root=$1
echo ${root}

nohup python -u construct_edges.py --input_file ${root}/data/sessions_train.csv \
  --output_file ${root}/n2v/edge.txt --log_file ${root}/log/n2v.log > log 2>&1 &

nohup python -u do_node2vec_walk.py --input_file ${root}/n2v/edge.txt --output_file ${root}/n2v/sentences.txt \
  --log_file ${root}/log/n2v.log > log 2>&1 &

nohup python -u word2vec.py --input_file ${root}/n2v/sentences.txt --output_file ${root}/n2v/n2v_emb.py \
  --log_file ${root}/log/n2v.log > log 2>&1 &