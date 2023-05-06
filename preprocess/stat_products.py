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


def stat(input_file):
    session_pd = pd.read_csv(input_file, sep=",")
    for index, row in tqdm(session_pd.iterrows(), desc="stat"):
        session = [sess.strip().strip("[").strip("]").strip("'") for sess in row["prev_items"].split()]
        next_item = row["next_item"]
        # session.append(next_item)
        session_set = set(session)


if __name__ == "__main__":
    logging.info(f"input_file:{config.input_file}")
    stat(config.input_file)