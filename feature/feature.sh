#!/bin/sh

root=$1
echo ${root}

nohup python -u item_stat_feature.py --input_file ${root}/data/sessions_train.csv \
  --output_file ${root}/feat/item_feat.dict > log 2>&1 &