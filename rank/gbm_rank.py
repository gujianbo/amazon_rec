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

    test_x = np.array([])
    test_y = np.array([])
    with open(test_file, "r") as fd:
        for line in tqdm(fd, desc="load test"):
            line = line.strip()
            (prev_items, candi, locale_code, item_feat_str, session_stat_feat_str, interact_feat_str,
             label) = line.split("\t")
            feat = []
            feat += [float(locale_code)]
            feat += [float(item) for item in item_feat_str.split(",")]
            feat += [float(item) for item in session_stat_feat_str.split(",")]
            feat += [float(item) for item in interact_feat_str.split(",")]
            test_x = np.append(test_x, np.array(feat))
            test_y = np.append(test_y, np.array([float(label)]))
    logging.info(f"test_x.shape:{test_x.shape}")
    train_data = lgb.Dataset(data=train_x, label=train_y)
    test_data = lgb.Dataset(data=test_x, label=test_y)
    num_round = 20
    param = {
        'learning_rate': 0.1,
        'max_depth': 5,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'num_trees': 100,
        'objective': 'binary',
        'metric': 'auc',
        'subsample': 0.7,
    }
    bst = lgb.train(param, train_data, num_round, valid_sets=[test_data])
    logging.info(f"train model done!")
    del train_x, train_y, test_x, test_y, train_data, test_data
    gc.collect()
    bst.save_model(model_file)
    logging.info(f"finish!")


if __name__ == "__main__":
    logging.info(f"train_file:{config.train_file}")
    logging.info(f"test_file:{config.test_file}")
    logging.info(f"model_file:{config.model_file}")
    train_gbm(config.train_file, config.test_file, config.model_file)
