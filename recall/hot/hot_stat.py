import os
import sys
sys.path.append("../..")
from utils.args import config
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd
import logging
from itertools import combinations
import itertools
from collections import Counter


def stat(input_file, output_file, topk):
    session_pd = pd.read_csv(input_file, sep=",")
    hot_dict = dict()

    for index, row in tqdm(session_pd.iterrows(), desc="gen_item_pair"):
        locale = row["locale"]
        session = [sess.strip().strip("[").strip("]").strip("'") for sess in row["prev_items"].split()]
        next_item = row["next_item"]
        seq_set = set(session)
        # seq_set.add(next_item)
        for item in seq_set:
            if locale not in hot_dict:
                hot_dict[locale] = Counter()
            hot_dict[locale][item] += 1

    top_hot_dict = dict()
    for country, cnt_dict in hot_dict.items():
        result = [k for k, v in cnt_dict.most_common(topk)]
        top_hot_dict[country] = result

    import pickle
    with open(output_file, 'wb') as fd:
        pickle.dump(top_hot_dict, fd)


if __name__ == "__main__":
    logging.info("input_file:" + config.input_file)
    logging.info("output_file:" + config.output_file)
    logging.info("topk:" + str(config.topk))
    stat(config.input_file, config.output_file, config.topk)
