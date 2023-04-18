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


def load_product(product_file):
    df = pd.read_csv(product_file)
    pro_dict = dict()
    for index, row in df:
        key = row["id"] + "_" + row["locale"]
        pro_dict[key] = row["title"]
    return pro_dict


def process(input_file, output_file, pro_dict):
    session_pd = pd.read_csv(input_file, sep=",")
    fd = open(output_file, "w")
    for index, row in tqdm(session_pd.iterrows(), desc="process"):
        session = [sess.strip().strip("[").strip("]").strip("'") for sess in row["prev_items"].split()]
        next_item = row["next_item"]
        locale = row["locale"]
        session.append(next_item)
        text = ""
        for item in session:
            key = item + "_" + locale
            title = ""
            if key in pro_dict:
                title = pro_dict[key]
            if text == "":
                text += title
            else:
                text += "[SEP]" + title
        fd.write(f"{text}\n")
    fd.close()


if __name__ == "__main__":
    print(f"input_file:{config.input_file}")
    print(f"output_file:{config.output_file}")
    print(f"product_file:{config.product_file}")
    pro_dict = load_product(config.product_file)
    process(config.input_file, config.output_file, pro_dict)
