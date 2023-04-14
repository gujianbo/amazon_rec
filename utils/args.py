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
parser.add_argument("--root_path", type=str, default='', help="root_path")
parser.add_argument("--log_file", type=str, default='', help="log_file")
parser.add_argument("--country", type=str, default='', help="country")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--topk', type=int, default=50)
parser.add_argument('--single_topk', type=int, default=30)

config = parser.parse_args()

config._PAD_ = 0