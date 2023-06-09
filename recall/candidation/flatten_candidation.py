import os
import sys
sys.path.append("../..")
from utils.args import config
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd
import logging
import random
import hashlib

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


def hash_code(x):
    return int(hashlib.md5((x.prev_items + str(config.seed)).encode('utf8')).hexdigest()[0:10], 16) % config.sample_cnt


def flatten(input_file, output_file, drop_no_hit, rate=0, if_hash_sample=0):
    df = pd.read_csv(input_file).drop(['Unnamed: 0'], axis=1)
    logging.info("load csv done!")
    if if_hash_sample == 1:
        df["hash"] = df.apply(hash_code, axis=1)
        df = df.loc[df.hash == 19]
        logging.info("hash sample done!")
    # 去除candidates没命中的行
    if drop_no_hit and if_hash_sample == 0:
        df["hit"] = df.apply(candi_in, axis=1)
        df = df.loc[df.hit == True].drop(['hit'], axis=1)
        logging.info("drop no hit done!")
    if rate > 0:
        df["sample_candi"] = df.apply(neg_sample, axis=1, args=(rate, ))
        df = df.drop(['candi'], axis=1)
        logging.info("sample done!")
        df_candi = df["sample_candi"].str.split(",", expand=True).stack().reset_index(level=1, drop=True)
        df_candi.name = 'recall_candi'
        df = df.drop(['sample_candi'], axis=1).join(df_candi)
        logging.info("expand done!")
    else:
        df_candi = df["candi"].str.split(",", expand=True).stack().reset_index(level=1, drop=True)
        df_candi.name = 'recall_candi'
        df = df.drop(['candi'], axis=1).join(df_candi)
        logging.info("expand done!")
    df['label'] = df.apply(tag, axis=1)
    logging.info("label tag done!")
    df = df.drop(['next_item'], axis=1)
    df.to_csv(output_file, index_label=None)


if __name__ == "__main__":
    logging.info(f"input_file:{config.input_file}")
    logging.info(f"output_file:{config.output_file}")
    logging.info(f"drop_no_hit:{config.drop_no_hit}")
    logging.info(f"neg_sample_rate:{config.neg_sample_rate}")

    flatten(config.input_file, config.output_file, config.drop_no_hit == 1, config.neg_sample_rate, config.if_hash_sample)
