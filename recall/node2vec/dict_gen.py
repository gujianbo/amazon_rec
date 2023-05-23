from tqdm.auto import tqdm
import cmath
import pandas as pd
import sys
sys.path.append("../..")
from utils.args import config

import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)


def dict_gen(input_file, output_file):
    i2i_dict = {}
    with open(input_file, "r") as f:
        for line in tqdm(f, desc="dict_gen"):
            item1, item2, score = line.strip().split("\t")
            i2i_dict.setdefault(item1, [])
            i2i_dict[item1].append(item2)
    import pickle
    with open(output_file, 'wb') as fd:
        pickle.dump(i2i_dict, fd)


if __name__ == "__main__":
    dict_gen(config.input_file, config.output_file)
