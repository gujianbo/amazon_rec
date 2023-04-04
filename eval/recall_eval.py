import os
import sys
sys.path.append("..")
from utils.args import config
import logging
from tqdm.auto import tqdm
import pandas as pd

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)


def load_recall(input_file, topk=100):
    i2i_dict = dict()
    with open(input_file, "r") as fd:
        last_item = ""
        item_cnt = 0
        for line in tqdm(fd, desc="load_recall"):
            left_item, right_item, score = line.split("\t")
            score = int(score)
            if last_item != left_item:
                item_cnt = 0

            if item_cnt < topk:
                if left_item not in i2i_dict:
                    i2i_dict[left_item] = []
                i2i_dict[left_item].append([right_item, score])
            item_cnt += 1
            last_item = left_item
    return i2i_dict


def eval(input_file, i2i_dict):
    session_pd = pd.read_csv(input_file, sep=",")

    cnt = 0
    hit_cnt10 = 0
    pre10_sum = 0
    recall10_sum = 0

    hit_cnt20 = 0
    pre20_sum = 0
    recall20_sum = 0

    hit_cnt50 = 0
    pre50_sum = 0
    recall50_sum = 0
    for index, row in tqdm(session_pd.iterrows(), desc="gen_item_pair"):
        session = [sess.strip().strip("[").strip("]").strip("'") for sess in row["prev_items"].split()]
        next_item = row["next_item"]

        candi_set10 = set()
        candi_set20 = set()
        candi_set50 = set()
        for aid in session:
            if aid in i2i_dict:
                top10 = set([item[0] for item in i2i_dict[aid][:10]])
                candi_set10 |= top10
                top20 = set([item[0] for item in i2i_dict[aid][:20]])
                candi_set20 |= top20
                top50 = set([item[0] for item in i2i_dict[aid][:50]])
                candi_set50 |= top50
        if next_item in candi_set10:
            hit_cnt10 += 1
            tmp_pre10 = 1/len(candi_set10)
            pre10_sum += tmp_pre10
            recall10_sum += 1

        if next_item in candi_set20:
            hit_cnt20 += 1
            tmp_pre20 = 1/len(candi_set20)
            pre20_sum += tmp_pre20
            recall20_sum += 1

        if next_item in candi_set50:
            hit_cnt50 += 1
            tmp_pre50 = 1/len(candi_set50)
            pre50_sum += tmp_pre50
            recall50_sum += 1
        cnt += 1

    pre10 = pre10_sum/cnt
    recall10 = recall10_sum/cnt
    hit_rate10 = hit_cnt10/cnt

    pre20 = pre20_sum/cnt
    recall20 = recall20_sum/cnt
    hit_rate20 = hit_cnt20/cnt

    pre50 = pre50_sum/cnt
    recall50 = recall50_sum/cnt
    hit_rate50 = hit_cnt50/cnt
    logging.info(f"precision@10:{pre10}, recall@10:{recall10}, hit_rate@10:{hit_rate10}")
    logging.info(f"precision@20:{pre20}, recall@20:{recall20}, hit_rate@20:{hit_rate20}")
    logging.info(f"precision@50:{pre50}, recall@50:{recall50}, hit_rate@50:{hit_rate50}")


if __name__ == "__main__":
    input_file = config.input_file
    i2i_file = config.i2i_file

    i2i_dict = load_recall(i2i_file, 50)
    eval(input_file, i2i_dict)
