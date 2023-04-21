import lightgbm as lgb
import os
import sys
sys.path.append("..")
from utils.args import config
from tqdm.auto import tqdm
import hashlib
import logging
import numpy as np
import gc

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)

def train_gbm(train_file, test_file, model_file):
    train_x = np.array([])
    train_y = np.array([])
    with open(train_file, "r") as fd:
        for line in tqdm(fd, desc="load train"):
            line = line.strip()
            (prev_items, candi, locale_code, item_feat_str, session_stat_feat_str, interact_feat_str,
             label) = line.split("\t")
            feat = []
            feat += [float(locale_code)]
            feat += [float(item) for item in item_feat_str.split(",")]
            feat += [float(item) for item in session_stat_feat_str.split(",")]
            feat += [float(item) for item in interact_feat_str.split(",")]
            train_x = np.append(train_x, np.array(feat))
            train_y = np.append(train_y, np.array([float(label)]))
    logging.info(f"train_x.shape:{train_x.shape}")
    train_data = lgb.Dataset(data=train_x, label=train_y)
    num_round = 20
    bst = lgb.train(param, train_data, num_round, valid_sets=[test_data])
    del train_x, train_y
    gc.collect()



if __name__ == "__main__":
    logging.info(f"train_file:{config.train_file}")
    logging.info(f"test_file:{config.test_file}")
    logging.info(f"model_file:{config.model_file}")
