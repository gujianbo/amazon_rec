#!/bin/sh

root=$1
echo ${root}

nohup python -u item_stat_feature.py --input_file ${root}/data/sessions_train.csv \
  --output_file ${root}/feat/item_feat.dict > log 2>&1 &


nohup python -u feat_extract.py --input_file ${root}/cand/candi_100_top100.pre.flatten.csv \
  --output_file ${root}/cand/candi_feat.pre \
  --item_feat_file ${root}/feat/item_feat.dict \
  --root_path ${root} > log 2>&1 &

nohup python -u feat_extract.py --input_file ${root}/cand/candi_100_top200.v5.flatten.csv \
  --output_file ${root}/cand/candi_100_top200.v5.feat \
  --item_feat_file ${root}/feat/item_feat.dict \
  --root_path ${root} > log 2>&1 &
nohup python -u feat_extract.py --input_file ${root}/cand/candi_100_top200.v5.flatten.csv \
  --output_file /home/zjlab/data/amazon/candi_100_top200.v5.feat \
  --item_feat_file ${root}/feat/item_feat.dict \
  --root_path ${root} > log 2>&1 &

# 测试数据
nohup python -u feat_extract.py --input_file ${root}/cand/candi_100_top100.pre.test.flatten.csv \
  --output_file ${root}/cand/candi_feat_100.pre.test \
  --item_feat_file ${root}/feat/item_feat.dict \
  --root_path ${root} > log 2>&1 &
# 测试数据
nohup python -u feat_extract.py --input_file ${root}/cand/candi_100_top200.v5.test.flatten.csv \
  --output_file ${root}/cand/candi_100_top200.v5.test \
  --item_feat_file ${root}/feat/item_feat.dict \
  --root_path ${root} > log 2>&1 &

# 提交数据
nohup python -u feat_extract.py --input_file ${root}/cand/submission_100_top100.flatten.csv \
  --output_file ${root}/cand/submission_feat_100 \
  --item_feat_file ${root}/feat/item_feat.dict \
  --root_path ${root} > log 2>&1 &

# 提交数据
nohup python -u feat_extract.py --input_file ${root}/cand/submission_100_top100.pre.flatten.csv \
  --output_file ${root}/cand/submission_feat_100.pre \
  --item_feat_file ${root}/feat/item_feat.dict \
  --root_path ${root} > log 2>&1 &
# 提交数据
nohup python -u feat_extract.py --input_file ${root}/cand/submission_100_top200.v5.flatten.csv \
  --output_file ${root}/cand/submission_100_top200.v5.feat \
  --item_feat_file ${root}/feat/item_feat.dict \
  --root_path ${root} > log 2>&1 &


nohup python -u feat_extract_dnn.py --input_file ${root}/cand/candi_100_top100.lo.flatten.csv \
  --output_file ${root}/cand/candi_feat.dnn \
  --item_feat_file ${root}/feat/item_feat.dict \
  --product_dict_file ${root}/data/product2id.dict \
  --root_path ${root} > log 2>&1 &

nohup python -u feat_extract_dnn.py --input_file ${root}/cand/candi_100_top100.lo.test.flatten.csv \
  --output_file ${root}/cand/candi_feat.dnn.test \
  --item_feat_file ${root}/feat/item_feat.dict \
  --product_dict_file ${root}/data/product2id.dict \
  --root_path ${root} > log 2>&1 &
