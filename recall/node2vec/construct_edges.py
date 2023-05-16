import sys
sys.path.append("../..")
from utils.args import config
from tqdm.auto import tqdm
import json
import logging
import pandas as pd

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)


def gen_item_edges(input_file, output_file):
    edges = dict()
    session_pd = pd.read_csv(input_file, sep=",")
    for index, row in tqdm(session_pd.iterrows(), desc="gen_item_edges"):
        session = [sess.strip().strip("[").strip("]").strip("'") for sess in row["prev_items"].split()]
        last_item = ""
        for item in session:
            if last_item == "":
                continue
            edges.setdefault(last_item, dict())
            edges[last_item].setdefault(item, 0)
            edges[last_item][item] += 1
            last_item = item

    fdout = open(output_file, "w")
    for left in edges.keys():
        for right in edges[left]:
            fdout.write(str(left)+" "+str(right)+" "+str(edges[left][right])+"\n")
    fdout.close()


if __name__ == "__main__":
    gen_item_edges(config.input_file, config.output_file)
