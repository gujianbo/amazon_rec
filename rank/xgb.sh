#!/bin/sh

root=$1
echo ${root}

nohup python -u slice_data.py --input_file ${root}/cand/candi_feat.dnn \
  --train_file ${root}/train_data/train --test_file ${root}/train_data/test \
  --sample_cnt 20 > log 2>&1 &

nohup python -u xgb_rank.py --train_file ${root}/train_data/train --test_file ${root}/train_data/test \
  --model_file ${root}/model/xgb.pre.1 --log_file ${root}/log/xgb.log --scale_pos_weight 10 \
  --num_boost_round 500 > log 2>&1 &

nohup python -u xgb_rank.py --train_file ${root}/train_data/train --test_file ${root}/train_data/test \
  --model_file ${root}/model/xgb.v5.1 --log_file ${root}/log/xgb.log --scale_pos_weight 10 \
  --num_boost_round 500 > log 2>&1 &

nohup python -u xgb_predict.py --input_file ${root}/cand/candi_feat_100.pre.test \
  --output_file ${root}/cand/candi_feat_100.pre.test.pred \
  --model_file ${root}/model/xgb.pre.1 > log 2>&1 &

# 提交数据
nohup python -u xgb_predict.py --input_file ${root}/cand/submission_feat_100.pre \
  --output_file ${root}/cand/submission_feat_100.pre.pred \
  --model_file ${root}/model/xgb.pre.1 > log 2>&1 &

# 格式化提交数据，输出parquet
python format_submission.py --input_file ${root}/cand/submission_feat_100.pre.pred \
  --output_file ${root}/cand/submission_100.pre.parquet

nohup python -u slice_data.py --input_file /home/zjlab/data/amazon/candi_100_top200.v5.feat \
  --train_file ${root}/train_data/train --test_file ${root}/train_data/test \
  --sample_cnt 20 > log 2>&1 &