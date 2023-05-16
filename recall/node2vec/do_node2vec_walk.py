import networkx as nx
import sys
from walker import RandomWalker
sys.path.append("../..")
from utils.args import config
import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)


if __name__ == "__main__":
    logging.info("loading edges...")
    G = nx.read_edgelist(config.input_file,
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    logging.info("loading edges finnish!")

    walker = RandomWalker(
        G, p=0.25, q=4, use_rejection_sampling=config.use_rejection_sampling == 1)

    logging.info("Preprocess transition probs...")
    logging.info("graph nodes length:" + str(len(G.nodes())))
    walker.preprocess_transition_probs()
    logging.info("Preprocess transition probs done!")
    logging.info("Start random walks...")
    walker.simulate_walks(
        num_walks=config.num_walks, walk_length=10, workers=1, verbose=1, sentence_file=config.output_file)

    logging.info("Finished!")