# -*- encoding: utf-8 -*-
import argparse

parser = argparse.ArgumentParser(description='Pipeline commandline argument')

# parameters for dataset settings
parser.add_argument("--input_file", type=str, default='', help="input_file")
parser.add_argument("--train_file", type=str, default='', help="train_file")
parser.add_argument("--test_file", type=str, default='', help="test_file")
parser.add_argument("--i2i_file", type=str, default='', help="i2i_file")
parser.add_argument("--output_file", type=str, default='', help="output_file")
parser.add_argument("--product_file", type=str, default='', help="product_file")
parser.add_argument("--product_dict_file", type=str, default='', help="product_dict_file")
parser.add_argument("--item_feat_file", type=str, default='', help="item_feat_file")
parser.add_argument("--model_file", type=str, default='', help="model_file")
parser.add_argument("--root_path", type=str, default='', help="root_path")
parser.add_argument("--log_file", type=str, default='', help="log_file")
parser.add_argument("--country", type=str, default='', help="country")
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--topk', type=int, default=50)
parser.add_argument('--single_topk', type=int, default=30)
parser.add_argument('--sample_cnt', type=int, default=5)
parser.add_argument('--is_train', type=int, default=1)
parser.add_argument('--if_hash_sample', type=int, default=0)

parser.add_argument('--drop_no_hit', type=int, default=1)
parser.add_argument('--neg_sample_rate', type=int, default=20)


#node2vec
parser.add_argument('--num_walks', type=int, default=60, help="num_walks")
parser.add_argument('--use_rejection_sampling', type=int, default=0, help="use_rejection_sampling")

# xgboost
parser.add_argument('--scale_pos_weight', type=int, default=20)
parser.add_argument('--num_boost_round', type=int, default=1000)

# dnn
parser.add_argument('--max_seq_len', type=int, default=64)
parser.add_argument('--d_model', type=int, default=64)
parser.add_argument('--dense_size', type=int, default=64)
parser.add_argument('--num_items', type=int, default=1420000)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--num_head', type=int, default=4)
parser.add_argument('--d_ff', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--init_parameters', type=str, default="")
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0, help="weight_decay")
parser.add_argument('--buffer_size', type=int, default=100000, help="buffer_size")
parser.add_argument('--train_batch_size', type=int, default=128, help="train_batch_size")
parser.add_argument('--test_batch_size', type=int, default=64, help="test_batch_size")
parser.add_argument('--epoch', type=int, default=30, help="epoch")
parser.add_argument('--log_level', type=int, default=0, help="log_level")
parser.add_argument('--log_interval', type=int, default=1000, help="log_interval")
parser.add_argument('--eval_step', type=int, default=10000, help="eval_step")
parser.add_argument('--save_step', type=int, default=10000, help="save_step")
parser.add_argument('--save_path', type=str, default="", help="save_path")
parser.add_argument('--step_lr_size', type=int, default=2, help="step_lr_size")

config = parser.parse_args()

config._PAD_ = 0