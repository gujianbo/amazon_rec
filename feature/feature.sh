#!/bin/sh

root=$1
echo ${root}

nohup python -u item_stat_feature.py --input_file ${root}/data/sessions_train.csv \
  --output_file ${root}/feat/item_feat.dict > log 2>&1 &


nohup python -u feat_extract.py --input_file ${root}/candi/candi_100_top100.flatten.csv \
  --output_file ${root}/candi/candi_feat \
  --item_feat_file ${root}/feat/item_feat.dict \
  --root_path ${root} > log 2>&1 &