

nohup python -u u2i_format.py --input_file ${root}/data/sessions_train.csv \
  --is_train 1 --train_file ${root}/cand/u2i_train --test_file ${root}/cand/u2i_test --sample_cnt 20 > log 2>&1 &


nohup python -u att2i_train.py --train_file ${root}/cand/ \
  --test_file ${root}/cand/test.123 --dense_size 220 --step_lr_size 1 \
  --save_path ${root}/models/ --log_file ${root}/log/dnn.log --weight_decay 0.01 --save_step 300000 > log 2>&1 &