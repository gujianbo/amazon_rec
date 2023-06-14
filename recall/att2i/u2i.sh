

nohup python -u u2i_format.py --input_file ${root}/data/sessions_train.csv \
  --is_train 1 --train_file ${root}/cand/u2i_train --test_file ${root}/cand/u2i_test \
  --product_dict_file ${root}/data/product2id.dict --sample_cnt 20  > log 2>&1 &


nohup python -u att2i_train.py --train_file ${root}/cand/u2i_train \
  --test_file ${root}/cand/u2i_test --step_lr_size 1 \
  --save_path ${root}/models/ --log_file ${root}/log/u2i.log \
  --d_model 128 --d_ff 128 \
  --weight_decay 0.01 --save_step 300000 > log 2>&1 &

nohup python -u att2i_item_inference.py \
  --test_file ${root}/data/product2id.dict --log_file ${root}/log/u2i.log --d_model 128 --d_ff 128 \
  --output_file ${root}/cand/att2i_item_vec \
  --init_parameters ${root}/models/u2i_v1686538789_steps_336240_4.1573.model > log 2>&1 &