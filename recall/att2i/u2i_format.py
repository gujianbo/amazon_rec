import os
import sys
sys.path.append("../..")
from utils.args import config
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd
import logging
import hashlib

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)


def map_id(prev_items, candi, product2id):
    assert candi in product2id
    candi_id = str(product2id[candi])
    prev_ids = []
    for item in prev_items:
        assert item in product2id
        prev_ids.append(str(product2id[item]))
    return ",".join(prev_ids), candi_id


def format(input_file, is_train=1, train_file=None, test_file=None, sample_cnt=None, product2id=None):
    session_pd = pd.read_csv(input_file, sep=",")
    train_fd = open(train_file, "w")
    if is_train == 1:
        test_fd = open(test_file, "w")

    for index, row in tqdm(session_pd.iterrows(), desc="gen_item_pair"):
        session = [sess.strip().strip("[").strip("]").strip("'") for sess in row["prev_items"].split()]
        if len(session) <= 3 and is_train == 1:
            continue

        locale = row["locale"]
        if is_train == 1:
            id_list = session[:-1]
            candi = session[-1]
        else:
            id_list = session
            candi = row["next_item"]

        id_list_feat, candi_id = map_id(id_list, candi, product2id)

        if is_train == 1:
            hash_val = int(hashlib.md5((id_list_feat + str(config.seed)).encode('utf8')).hexdigest()[0:10], 16) % sample_cnt
            if hash_val == 0:
                test_fd.write(f"{id_list_feat}\t{candi_id}\t{locale}\n")
            else:
                train_fd.write(f"{id_list_feat}\t{candi_id}\t{locale}\n")
        else:
            train_fd.write(f"{id_list_feat}\t{candi_id}\t{locale}\n")


if __name__ == "__main__":
    logging.info("input_file:" + config.input_file)
    logging.info("train_file:" + config.train_file)
    logging.info("test_file:" + config.test_file)
    logging.info("is_train:" + str(config.is_train))
    logging.info("product2id:" + str(config.product2id))

    import pickle
    product2id = pickle.load(open(config.product_dict_file, 'rb'))

    format(config.input_file, config.is_train, config.train_file, config.test_file, config.sample_cnt, config.product2id)
