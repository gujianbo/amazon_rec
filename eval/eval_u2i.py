import os
import sys
sys.path.append("..")
from utils.args import config
import logging
from tqdm.auto import tqdm
import pandas as pd

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)


def eval(input_file):
    fd = open(input_file, "r")
    cnt = 0
    hit = 0
    for line in tqdm(fd, desc="eval"):
        cnty, next_item, cand = line.strip().split("\t")
        cand_set = set(cand.split(","))
        if next_item in cand_set:
            hit += 1
        cnt += 1
    hit_rate = hit/cnt
    logging.info(f"{input_file} - hit_rate@10:{hit_rate}")


if __name__ == "__main__":
    input_file = config.input_file

    eval(input_file)
