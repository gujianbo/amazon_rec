#!/bin/sh

root=$1
echo ${root}

nohup python -u construct_edge.py --input_file ${root}/data/sessions_train.csv \
  --output_file ${root}/n2v/edge.txt --log ${root}/log/n2v.log > log 2>&1 &

python -u do_node2vec_walk.py --input_file ${root}/n2v/edge.txt --output_file ${root}/n2v/sentences.txt

python -u word2vec.py