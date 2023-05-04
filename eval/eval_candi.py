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
    hit100cnt = 0
    hit150cnt = 0
    hit200cnt = 0
    mrr100 = 0
    hit_cnt = 0
    for index, row in tqdm(session_pd.iterrows(), desc="eval_candi"):
        next_item = row["next_item"]
        candi = row["candi"].split(",")
        # if next_item in candi:
        #     hit200cnt += 1
        if next_item in candi[:10]:
            hit10cnt += 1
        if next_item in candi[:20]:
            hit20cnt += 1
        if next_item in candi[:50]:
            hit50cnt += 1
        if next_item in candi[:100]:
            hit100cnt += 1
            idx = candi[:100].index(next_item) + 1
            mrr100 += 1.0/idx

        if next_item in candi[:150]:
            hit150cnt += 1
        if next_item in candi[:200]:
            hit200cnt += 1
        if next_item in candi:
            hit_cnt += 1
        # if next_item in candi[:150]:
        #     hit150cnt += 1
        cnt += 1

    hit10rate = hit10cnt / cnt
    hit20rate = hit20cnt / cnt
    hit50rate = hit50cnt / cnt
    hit100rate = hit100cnt / cnt
    hit150rate = hit150cnt / cnt
    hit200rate = hit200cnt / cnt
    hit_rate = hit_cnt / cnt
    mrr100avg = mrr100/cnt
    logging.info(f"{input_file} - hit_rate@10:{hit10rate}")
    logging.info(f"{input_file} - hit_rate@20:{hit20rate}")
    logging.info(f"{input_file} - hit_rate@50:{hit50rate}")
    logging.info(f"{input_file} - hit_rate@100:{hit100rate}")
    logging.info(f"{input_file} - hit_rate@150:{hit150rate}")
    logging.info(f"{input_file} - hit_rate@200:{hit200rate}")
    logging.info(f"{input_file} - hit_rate:{hit_rate}")
    logging.info(f"mmr@100:{mrr100avg}")


if __name__ == "__main__":
    input_file = config.input_file
    eval_candi(input_file)
