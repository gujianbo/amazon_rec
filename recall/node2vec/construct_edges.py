import sys
sys.path.append("../..")
from utils.args import config
from tqdm.auto import tqdm
import json
import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)


def gen_item_edges(input_file, output_file, edge_time_span, debug=0):
    edges = dict()
    with open(input_file, "r") as f:
        for line in tqdm(f, desc="gen_item_edges"):
            session = json.loads(line.strip())
            last_ts = 0
            last_aid = 0
            for event in session['events']:
                cur_ts = event["ts"]
                cur_aid = event["aid"]
                if last_ts != 0 and cur_ts - last_ts < edge_time_span:
                    edges.setdefault(last_aid, dict())
                    edges[last_aid].setdefault(cur_aid, 0)
                    edges[last_aid][cur_aid] += 1

                last_ts = cur_ts
                last_aid = cur_aid

    fdout = open(output_file, "w")
    for left in edges.keys():
        for right in edges[left]:
            fdout.write(str(left)+" "+str(right)+" "+str(edges[left][right])+"\n")
    fdout.close()


if __name__ == "__main__":
    gen_item_edges(config.input_file, config.output_file, config.edge_time_span)
