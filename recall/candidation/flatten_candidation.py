import os
import sys
sys.path.append("../..")
from utils.args import config
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd
import logging
import random

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)


def candi_in(x):
    return x.next_item in x.candi.split(",")


def neg_sample(x, rate):
    all_candi = x.candi.split(",")
    if len(all_candi) > rate:
        sample_candi = random.sample(all_candi, rate)
        if x.next_item not in sample_candi:
            sample_candi.append(x.next_item)
        return ",".join(sample_candi)
    else:
        return x.candi


def tag(x):
    if x.next_item == x.recall_candi:
        return 1
    else:
        return 0


def flatten(input_file, output_file, drop_no_hit, rate=0):
    df = pd.read_csv(input_file).drop(['Unnamed: 0'], axis=1)
    # 去除candidates没命中的行
    if drop_no_hit:
        df["hit"] = df.apply(candi_in, axis=1)
        df = df.loc[df.hit == True].drop(['hit'], axis=1)
    if rate > 0:
        df["sample_candi"] = df.apply(neg_sample, axis=1, args=(rate, ))
        df = df.drop(['candi'], axis=1)
        df_candi = df["sample_candi"].str.split(",", expand=True).stack().reset_index(level=1,drop=True)
        df_candi.name = 'recall_candi'
        df = df.drop(['sample_candi'], axis=1).join(df_candi)
    else:
        df_candi = df["candi"].str.split(",", expand=True).stack().reset_index(level=1, drop=True)
        df_candi.name = 'recall_candi'
        df = df.drop(['candi'], axis=1).join(df_candi)
    df['label'] = df.apply(tag, axis=1)
    df = df.drop(['next_item'], axis=1)
    df.to_csv(output_file, index_label=None)


if __name__ == "__main__":
    logging.info("input_file:" + config.input_file)
    logging.info("output_file:" + config.output_file)

    flatten(config.input_file, config.output_file)
