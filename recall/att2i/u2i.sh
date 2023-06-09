

nohup python -u u2i_format.py --input_file ${root}/data/sessions_train.csv \
  --is_train 1 --train_file ${root}/candi/u2i_train --test_file ${root}/candi/u2i_test --sample_cnt 20 > log 2>&1 &