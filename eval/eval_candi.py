import os
import sys
sys.path.append("..")
from utils.args import config
import logging
from tqdm.auto import tqdm
import pandas as pd

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)

def eval_candi(input_file):
    session_pd = pd.read_csv(input_file, sep=",")
    cnt = 0
    hit10cnt = 0
    hit20cnt = 0
    hit50cnt = 0
    for index, row in tqdm(session_pd.iterrows(), desc="eval_candi"):
        next_item = row["next_item"]
        candi = row["candi"].split()
        if next_item in candi:
            hit50cnt += 1
        if next_item in candi[:10]:
            hit10cnt += 1
        if next_item in candi[:20]:
            hit20cnt += 1
        cnt += 1

    hit50rate = hit50cnt / cnt
    hit10rate = hit10cnt / cnt
    hit20rate = hit20cnt / cnt
    logging.info(f"hit_rate@10:{hit10rate}")
    logging.info(f"hit_rate@20:{hit20rate}")
    logging.info(f"hit_rate@50:{hit50rate}")


if __name__ == "__main__":
    input_file = config.input_file
    eval_candi(input_file)
