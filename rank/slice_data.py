
import os
import sys
sys.path.append("..")
from utils.args import config
from tqdm.auto import tqdm
import hashlib
import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)


def slice_data(input_file, train_file, test_file, sample_cnt, seed):
    fdout_train = open(train_file, "w")
    fdout_test = open(test_file, "w")
    with open(input_file, "r") as fd:
        last_prev_items = ""
        buf = []
        for line in tqdm(fd, ):
            line = line.strip()
            (prev_items, candi, locale_code, item_feat_str, session_stat_feat_str, interact_feat_str, label) = line.split("\t")
            if last_prev_items == "" or prev_items == last_prev_items:
                buf.append(line)
                last_prev_items = prev_items
                continue
            hash_val = int(hashlib.md5((last_prev_items + str(seed)).encode('utf8')).hexdigest()[0:10], 16) % sample_cnt
            for item in buf:
                if hash_val == 0:
                    fdout_test.write(item+"\n")
                else:
                    fdout_train.write(item+"\n")
            buf = []
            buf.append(line)
            last_prev_items = prev_items
        hash_val = int(hashlib.md5((last_prev_items + str(seed)).encode('utf8')).hexdigest()[0:10], 16) % sample_cnt
        for item in buf:
            if hash_val == 0:
                fdout_test.write(item + "\n")
            else:
                fdout_train.write(item + "\n")


if __name__ == "__main__":
    logging.info(f"input_file:{config.input_file}")
    logging.info(f"train_file:{config.train_file}")
    logging.info(f"test_file:{config.test_file}")
    logging.info(f"sample_cnt:{config.sample_cnt}")
    logging.info(f"seed:{config.seed}")
    slice_data(config.input_file, config.train_file, config.test_file, config.sample_cnt, config.seed)