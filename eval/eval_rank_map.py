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
    buf = []
    last_prev_items = ""
    map_val = 0
    sum = 0
    with open(input_file, "r") as fd:
        for line in tqdm(fd, desc="eval"):
            (prev_items, candi, locale_code, label, pred) = line.strip().split("\t")
            if last_prev_items != "" and prev_items != last_prev_items:
                sorted_buf = sorted(buf, key=lambda x: x[2], reverse=True)
                for i in range(len(sorted_buf)):
                    candi, label, pred = sorted_buf[i]
                    if label == 1:
                        map_val += 1.0/(i+1)
                        break
                sum += 1
                buf = []
            buf.append([candi, int(label), float(pred)])

            last_prev_items = prev_items

    sorted_buf = sorted(buf, key=lambda x: x[2], reverse=True)
    for i in range(len(sorted_buf)):
        candi, label, pred = sorted_buf[i]
        if label == 1:
            map_val += 1.0 / (i + 1)
            break
    sum += 1
    map_avg = map_val/sum
    logging.info(f"MAP value:{map_avg}")


if __name__ == "__main__":
    logging.info(f"input_file:{config.input_file}")
    eval(config.input_file)


