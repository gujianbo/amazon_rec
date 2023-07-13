

nohup python -u u2i_format.py --input_file ${root}/data/sessions_train.csv \
  --is_train 1 --train_file ${root}/cand/u2i_train --test_file ${root}/cand/u2i_test \
  --product_dict_file ${root}/data/product2id.dict --sample_cnt 20  > log 2>&1 &


nohup python -u att2i_train.py --train_file ${root}/cand/u2i_train \
  --test_file ${root}/cand/u2i_test --step_lr_size 1 \
  --save_path ${root}/models/ --log_file ${root}/log/u2i.log \
  --d_model 128 --d_ff 128 --temperature 0.1 --train_batch_size 512 \
  --epoch 60 --lr 1e-3 --step_lr_size 5 \
  --weight_decay 0.01 --save_step 50000 > log 2>&1 &

nohup python -u att2i_train.py --train_file ${root}/cand/u2i_train \
  --test_file ${root}/cand/u2i_test \
  --save_path ${root}/models/ --log_file ${root}/log/u2i.log \
  --d_model 128 --d_ff 128 --temperature 0.1 --train_batch_size 512 \
  --epoch 60 --lr 1e-3 --step_lr_size 5 --eval_step 5000 \
  --weight_decay 3.0 --save_step 50000 > log 2>&1 &

nohup python -u att2i_item_inference.py \
  --test_file ${root}/data/product2id.dict --log_file ${root}/log/u2i.log --d_model 128 --d_ff 128 \
  --output_file ${root}/cand/att2i_item_vec_v1688543483 \
  --init_parameters ${root}/models/u2i_v1688543483_s28000_0.1_0.001_2.6_512_5.7405.model > log 2>&1 &

nohup python -u att2i_user_inference.py \
  --test_file ${root}/cand/u2i_test --log_file ${root}/log/u2i.log --d_model 128 --d_ff 128 \
  --output_file ${root}/cand/att2i_user_test_vec \
  --init_parameters ${root}/models/u2i_v1686538789_steps_336240_4.1573.model > log 2>&1 &

nohup python -u att2i_topk.py \
  --product_file ${root}/data/products_train.csv --itemvec_file ${root}/cand/att2i_item_vec --idproduct_dict_file ${root}/data/id2product.dict \
  --item_file ${root}/cand/part_item_vec --input_file ${root}/cand/att2i_user_test_vec --output_file ${root}/cand/att2i_user_test_topk --dim 64 \
  --batch_size 128 --use_gpu 1 > log 2>&1 &