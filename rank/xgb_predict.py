import xgboost
import os
import sys
sys.path.append("..")
from utils.args import config
from tqdm.auto import tqdm
import hashlib
import logging
import numpy as np
import pandas as pd
import glob

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)


def predict_test(input_file, output_file, model_file, buf_len=1000):
    train_x_arr = []
    buf = []
    model_xgb = xgboost.Booster()
    model_xgb.load_model(model_file)
    fdout = open(output_file, "w")
    with open(input_file, "r") as fd:
        for line in tqdm(fd, desc="load train"):
            line = line.strip()
            # (prev_items, candi, locale_code, item_feat_str, session_stat_feat_str, interact_feat_str,
            #  label) = line.split("\t")
            (prev_items, candi, locale_code, item_feat_str, session_stat_feat_str, interact_feat_str,
             local_sec_feat_str, locale_code_feat_str, label) = line.split("\t")
            if locale_code in ['4', '5', '6']:
                continue
            if len(buf) >= buf_len:
                train_x = np.array(train_x_arr)
                train_x = xgboost.DMatrix(data=train_x)
                pred_y = model_xgb.predict(train_x)
                for i in range(len(buf)):
                    tmp_prev_items, tmp_candi, tmp_locale_code, tmp_label = buf[i]
                    score = pred_y[i]
                    fdout.write(f"{tmp_prev_items}\t{tmp_candi}\t{tmp_locale_code}\t{tmp_label}\t{score}\n")
                buf = []
                train_x_arr = []
            # feat = []
            # feat += [float(locale_code)]
            # feat += [float(item) for item in item_feat_str.split(",")]
            # feat += [float(item) for item in session_stat_feat_str.split(",")]
            # feat += [float(item) for item in interact_feat_str.split(",")]
            feat = []
            # feat += [float(locale_code)]
            feat += [float(item) for item in item_feat_str.split(",")]
            feat += [float(item) for item in session_stat_feat_str.split(",")]
            feat += [float(item) for item in interact_feat_str.split(",")]
            feat += [float(item) for item in local_sec_feat_str.split(",")]
            feat += [float(item) for item in locale_code_feat_str.split(",")]
            assert len(feat) == 344

            train_x_arr.append(feat)
            buf.append([prev_items, candi, locale_code, float(label)])

        train_x = np.array(train_x_arr)
        train_x = xgboost.DMatrix(data=train_x)
        pred_y = model_xgb.predict(train_x)
        for i in range(len(buf)):
            tmp_prev_items, tmp_candi, tmp_locale_code, tmp_label = buf[i]
            score = pred_y[i]
            fdout.write(f"{tmp_prev_items}\t{tmp_candi}\t{tmp_locale_code}\t{tmp_label}\t{score}\n")
    fdout.close()


if __name__ == "__main__":
    logging.info(f"input_file:{config.input_file}")
    logging.info(f"output_file:{config.output_file}")
    logging.info(f"model_file:{config.model_file}")
    predict_test(config.input_file, config.output_file, config.model_file)
