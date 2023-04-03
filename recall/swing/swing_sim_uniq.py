import os
import sys
sys.path.append("../..")
from utils.args import config
from tqdm.auto import tqdm
from utils.heap import Heap
import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)


def uniq_sim(out_path):
    infile = out_path + ".full"
    item_sim_dict = dict()
    logging.info("uniq_sim file:" + infile)
    with open(infile, "r") as f:
        for item_pair in tqdm(f, desc="uniq_sim:"+infile):
            pair_str = item_pair.strip()
            item_i, item_j, score = pair_str.split("\t")
            item_i = int(item_i)
            item_j = int(item_j)
            score = float(score)
            item_sim_dict.setdefault(item_i, Heap(item_i, 100))
            item_sim_dict[item_i].enter_item([item_j, score])

            item_sim_dict.setdefault(item_j, Heap(item_j, 100))
            item_sim_dict[item_j].enter_item([item_i, score])

    return item_sim_dict


def output_sim_file(item_sim_dict, out_path):
    fd = open(out_path, "w")
    for item, sim_items in item_sim_dict.items():
        sim_score = sim_items.top_items()
        for (item_j, score) in sim_score:
            fd.write(str(item) + "\t" + str(item_j) + "\t" + str(score) + "\n")
    fd.close()


if __name__ == "__main__":
    item_sim_dict = uniq_sim(config.output_file)
    output_sim_file(item_sim_dict, config.output_file)