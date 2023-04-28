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

parser.add_argument('--drop_no_hit', type=int, default=1)
parser.add_argument('--neg_sample_rate', type=int, default=20)

# xgboost
parser.add_argument('--scale_pos_weight', type=int, default=20)
parser.add_argument('--num_boost_round', type=int, default=1000)

config = parser.parse_args()

config._PAD_ = 0