
import os
import sys
sys.path.append("../..")
from utils.args import config
from tqdm.auto import tqdm
import json
from pathlib import Path
import random
import pandas as pd
from copy import deepcopy
from utils.heap import Heap
import logging
from itertools import combinations
import hashlib

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)

item_sim_dict = dict()

def load_session2item(session_file):
    session2item = dict()
    with open(session_file, "r") as f:
        for line in tqdm(f, desc="load_session2item"):
            line = line.strip()
            session, items = line.split("\t")
            session = int(session)
            item_set = set(map(int, items.split(",")))
            session2item[session] = item_set
    return session2item


def load_item2session(i, j, output_file):
    item2session = dict()
    item_path = Path(output_file).parent.joinpath("item")
    with open(str(item_path) + "/item2session_" + str(i), "r") as f:
        for line in tqdm(f, desc="load_item2session_" + str(i)):
            line = line.strip()
            item, sessions = line.split("\t")
            # item = int(item)
            session_set = set(sessions.split(","))
            item2session[item] = session_set

    with open(str(item_path) + "/item2session_" + str(j), "r") as f:
        for line in tqdm(f, desc="load_item2session_" + str(j)):
            line = line.strip()
            item, sessions = line.split("\t")
            # item = int(item)
            session_set = set(sessions.split(","))
            item2session[item] = session_set
    return item2session


def calc_simlarity(session_item, i, j, output_file, sim_fd, alpha=1.0, session_num_threhold=10000, debug=0):
    item2session = load_item2session(i, j, output_file)
    logging.info("load_item2session "+str(i) + "_" + str(j) + " done!")
    pair_path = Path(output_file).parent.joinpath("pair")

    pair_file = str(pair_path) + "/uniq_pair_" + str(i) + "_" + str(j)
    idx = 0
    logging.info("calc_simlarity file " + pair_file)
    with open(pair_file, "r") as f:
        for item_pair in tqdm(f, desc="calc_simlarity:"+pair_file):
            pair_str = item_pair.strip()
            item_i, item_j = pair_str.split(",")
            # item_i = int(item_i)
            # item_j = int(item_j)
            common_sessions = item2session[item_i] & item2session[item_j]
            if len(common_sessions) <= 1:
                continue
            # 采个样，防止太多，撑爆内存
            elif len(common_sessions) > session_num_threhold:
                common_sessions = random.sample(common_sessions, session_num_threhold)
            session_pairs = list(combinations(common_sessions, 2))
            result = 0.0
            for (user_u, user_v) in session_pairs:
                result += 1 / (alpha + len(session_item[user_u] & session_item[user_v]))

            # item_sim_dict.setdefault(item_i, Heap(item_i, 100))
            # item_sim_dict[item_i].enter_item([item_j, result])
            #
            # item_sim_dict.setdefault(item_j, Heap(item_j, 100))
            # item_sim_dict[item_j].enter_item([item_i, result])
            sim_fd.write(str(item_i)+"\t"+str(item_j)+"\t"+str(result)+"\n")

            if debug == 1:
                idx += 1
                if idx >= 1000:
                    break
    del item2session
    logging.info("calc_simlarity file " + pair_file + " done!")


# def output_sim_file(item_sim_dict, out_path):
#     fd = open(out_path, "w")
#     for item, sim_items in item_sim_dict.items():
#         sim_score = sim_items.top_items()
#         for (item_j, score) in sim_score:
#             fd.write(str(item) + "\t" + str(item_j) + "\t" + str(score) + "\n")
#     fd.close()


def main(debug):
    logging.info("output_file:" + str(debug))
    logging.info("output_file:" + config.output_file)

    session_file = os.path.abspath(os.path.join(config.output_file, "..")) + "/session/session2item"
    logging.info("To load " + str(session_file))
    session2item = load_session2item(str(session_file))
    logging.info("load_session2item done!")

    sim_fd = open(config.output_file+".full", "w")
    idx = 0
    for i in range(10):
        for j in range(10):
            calc_simlarity(session2item, i, j, config.output_file, sim_fd, debug=debug)
            if debug == 1:
                idx += 1
                if idx == 1:
                    break
        if debug == 1:
            if idx == 1:
                break
    sim_fd.close()
    logging.info("Finish!")


if __name__ == "__main__":
    main(config.debug)
