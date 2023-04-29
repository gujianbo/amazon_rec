import os
import sys
sys.path.append("..")
from utils.args import config
import logging
from tqdm.auto import tqdm
import pandas as pd

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)


def format(input_file, output_file):
    buf = []
    last_prev_items = ""

    out_dict = {"next_item_prediction": []}
    with open(input_file, "r") as fd:
        for line in tqdm(fd, desc="format"):
            (prev_items, candi, locale_code, label, pred) = line.strip().split("\t")
            if last_prev_items != "" and prev_items != last_prev_items:
                sorted_buf = sorted(buf, key=lambda x: x[1], reverse=True)
                # print(sorted_buf)
                items = [item[0] for item in sorted_buf]
                out_dict["next_item_prediction"].append(items)
                buf = []
            buf.append([candi, float(pred)])

            last_prev_items = prev_items
    sorted_buf = sorted(buf, key=lambda x: x[1], reverse=True)
    # print(sorted_buf)
    items = [item[0] for item in sorted_buf]
    out_dict["next_item_prediction"].append(items)
    out_pd = pd.DataFrame(out_dict)
    print(out_pd)
    out_pd.to_parquet(output_file)


if __name__ == "__main__":
    logging.info(f"input_file:{config.input_file}")
    logging.info(f"output_file:{config.output_file}")
    format(config.input_file, config.output_file)