#!/bin/sh

root=$1
echo ${root}

# 对计算的相似度进行过滤，保留每个topk
python -u recall_eval.py --input_file ${root}/data/sessions_train.csv --i2i_file ${root}/swing/swing.sim \
  --log_file ${root}/log/eval.log

root="/data/users/jianbogu/amazon/"
python -u recall_eval.py --input_file ${root}/data/sessions_train.csv --i2i_file ${root}/covisit/co.global.txt \
  --log_file ${root}/log/eval1.log

python -u eval_candi.py --input_file /data/users/jianbogu/amazon/cand/candi_100_top200.lo.csv \
  --log_file ${root}/log/eval_candi2.log

python -u eval_candi.py --input_file /data/users/jianbogu/amazon/cand/candi_100_all.sample.lo.csv \
  --log_file ${root}/log/eval_candi2.log

nohup python -u eval_rank_mrr.py --input_file ${root}/cand/candi_feat_100.pre.test.pred \
  --log_file ${root}/log/eval_rank_mrr.log > log 2>&1 &

nohup python -u eval_rank_mrr.py --input_file ${root}/cand/candi_feat.dnn.test.pred  \
  --log_file ${root}/log/eval_rank_mrr.log > log 2>&1 &

python -u eval_candi.py --input_file /data/users/jianbogu/amazon/cand/candi_100_top100.v4.csv \
  --log_file ${root}/log/eval_candi2.log