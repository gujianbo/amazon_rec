#!/bin/sh

root=$1
echo ${root}

nohup python -u slice_data.py --input_file ${root}/cand/candi_feat_10 \
  --train_file ${root}/train_data/train --test_file ${root}/train_data/test \
  --sample_cnt 20 > log 2>&1 &

nohup python -u xgb_rank.py --train_file ${root}/train_data/train --test_file ${root}/train_data/test \
  --model_file ${root}/model/xgb.1 --log_file ${root}/log/xgb.log --scale_pos_weight 10 \
  --num_boost_round 500 > log 2>&1 &

nohup python -u xgb_predict.py --input_file ${root}/cand/candi_feat_100.test --output_file ${root}/cand/candi_feat_100.test.pred \
  --model_file ${root}/model/xgb.1 > log 2>&1 &