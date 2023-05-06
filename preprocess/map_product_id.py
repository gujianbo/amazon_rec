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

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)


def map_id(input_file, output_file):
    df = pd.read_csv(input_file)
    id2product = {}
    product2id = {}
    for index, row in tqdm(df.iterrows(), desc="stat"):
        product_id = row["id"]
        id = len(product2id) + 10
        if product_id not in product2id:
            product2id[product_id] = id
            id2product[id] = product_id
    print(f"product length：{len(product2id)}, {len(id2product)}")

    import pickle
    with open(output_file + "product2id.dict", 'wb') as fd:
        pickle.dump(product2id, fd)
    with open(output_file + "id2product.dict", 'wb') as fd:
        pickle.dump(id2product, fd)


if __name__ == "__main__":
    logging.info(f"input_file:{config.input_file}")
    map_id(config.input_file)