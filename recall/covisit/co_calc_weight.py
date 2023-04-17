import os
import sys
sys.path.append("../..")
from utils.args import config
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd
import logging
from itertools import combinations
import hashlib

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)


def calc_weight(input_file, output_file, topk):
    pair_df = pd.read_csv(input_file, sep=",")
    pair_df = pair_df.sort_values(['item1', 'weight'], ascending=[True, False])
    pair_df = pair_df.reset_index()
    pair_df['n'] = pair_df.groupby('item1').item2.cumcount()
    pair_df = pair_df.loc[pair_df.n < topk].drop('n', axis=1)

    co_dict = pair_df.groupby('item1').item2.apply(list)
    import pickle
    with open(output_file, 'wb') as fd:
        pickle.dump(co_dict.to_dict(), fd)


if __name__ == "__main__":
    logging.info("input_file:" + config.input_file)
    logging.info("output_file:" + config.output_file)
    logging.info("topk:" + config.topk)
    calc_weight(config.input_file, config.output_file, config.topk)
