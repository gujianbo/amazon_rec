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


def calc_weight(input_file, output_file):
    pair_df = pd.read_csv(input_file, sep=",")
    pair_df = pair_df.sort_values(['item1', 'weight'], ascending=[True, False])
    pair_df = pair_df.reset_index()
    pair_df['n'] = pair_df.groupby('item1').item2.cumcount()
    pair_df = pair_df.loc[pair_df.n < 100].drop('n', axis=1)
    with open(output_file, "w") as fd:
        for row in pair_df.iterrows():
            fd.write(f"{row['item1']}\t{row['item2']}\t{row['score']}\n")


if __name__ == "__main__":
    logging.info("input_file:" + config.input_file)
    logging.info("output_file:" + config.output_file)
    calc_weight(config.input_file, config.output_file, config.debug)
