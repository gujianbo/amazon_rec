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
    mmr_val = 0
    sum = 0
    with open(input_file, "r") as fd:
        for line in tqdm(fd, desc="eval"):
            (prev_items, candi, locale_code, label, pred) = line.strip().split("\t")
            if last_prev_items != "" and prev_items != last_prev_items:
                sorted_buf = sorted(buf, key=lambda x: x[2], reverse=True)
                print(sorted_buf)
                items = ",".join([item[0] for item in sorted_buf])
                target = ""
                for i in range(len(sorted_buf)):
                    tmp_candi, tmp_label, tmp_pred = sorted_buf[i]
                    if tmp_label == 1.0:
                        mmr_val += 1.0/(i+1.0)
                        target = tmp_candi
                        break
                logging.info(f"{last_prev_items}\t{items}\t{target}")
                sum += 1
                buf = []
            buf.append([candi, float(label), float(pred)])

            last_prev_items = prev_items

    sorted_buf = sorted(buf, key=lambda x: x[2], reverse=True)
    items = ",".join([item[0] for item in sorted_buf])
    for i in range(len(sorted_buf)):
        tmp_candi, tmp_label, tmp_pred = sorted_buf[i]
        if tmp_label == 1:
            mmr_val += 1.0 / (i + 1.0)
            target = tmp_candi
            break
    logging.info(f"{last_prev_items}\t{items}\t{target}")
    sum += 1
    mmr_avg = mmr_val/sum
    logging.info(f"mmr_val value:{mmr_val}")
    logging.info(f"sum:{sum}")
    logging.info(f"MMR value:{mmr_avg}")


if __name__ == "__main__":
    logging.info(f"input_file:{config.input_file}")
    eval(config.input_file)


