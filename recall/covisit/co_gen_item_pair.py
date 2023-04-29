import os
import sys
sys.path.append("../..")
from utils.args import config
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd
import logging
from itertools import combinations
import hashlib

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)


def gen_item_pair(input_file, output_file, country, debug=0):
    # session_item = dict()
    pair_path = Path(output_file).parent

    fdout = open(str(pair_path)+"/pairs." + country + ".csv", "w")
    fdout.write("item1,item2,weight\n")
    idx = 0

    session_pd = pd.read_csv(input_file, sep=",")
    if country != "global":
        session_pd = session_pd[session_pd.locale == country]

    for index, row in tqdm(session_pd.iterrows(), desc="gen_item_pair"):
        session = [sess.strip().strip("[").strip("]").strip("'") for sess in row["prev_items"].split()]
        next_item = row["next_item"]
        # session.append(next_item)
        session_set = set(session)
        # print(click_aid)
        if len(session_set) <= 1:
            continue
        pair_list = list(combinations(session_set, 2))

        for pair in pair_list:
            if pair[0] == pair[1]:
                continue
            idx_1 = session.index(pair[0])
            idx_2 = session.index(pair[1])
            weight = (1/2)**(abs(idx_1-idx_2))

            hash1 = int(hashlib.md5((str(pair[0]) + str(config.seed)).encode('utf8')).hexdigest()[0:10], 16) % 10
            hash2 = int(hashlib.md5((str(pair[1]) + str(config.seed)).encode('utf8')).hexdigest()[0:10], 16) % 10
            hash_code = 10 * hash1 + hash2
            fdout.write(f"{pair[0]},{pair[1]},{weight}\n")
            fdout.write(f"{pair[1]},{pair[0]},{weight}\n")
        if debug == 1:
            idx += 1
            if idx >= 10000:
                break
    fdout.close()


if __name__ == "__main__":
    logging.info("input_file:" + config.input_file)
    logging.info("output_file:" + config.output_file)
    logging.info(f"country:{config.country}")
    gen_item_pair(config.input_file, config.output_file, config.country, config.debug)

