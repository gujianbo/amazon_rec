from gensim.models import Word2Vec
from tqdm.auto import tqdm

import sys
sys.path.append("../..")
from utils.args import config
import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)


def train(sentences, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):
    kwargs["sentences"] = sentences
    kwargs["min_count"] = kwargs.get("min_count", 0)
    kwargs["vector_size"] = embed_size
    kwargs["sg"] = 1
    kwargs["hs"] = 0  # node2vec not use Hierarchical Softmax
    kwargs["workers"] = workers
    kwargs["window"] = window_size
    kwargs["epochs"] = iter

    logging.info("Learning embedding vectors...")
    model = Word2Vec(**kwargs)
    logging.info("Learning embedding vectors done!")

    return model


def get_embeddings(model, words_set, output_file):
    if model is None:
        print("model not train")
        return {}

    with open(output_file, "w") as f:
        for word in words_set:
            try:
                emb = ','.join(map(str, model.wv[word]))
                f.write(str(word) + "\t" + emb + "\n")
            except:
                logging.error("word " + word + " not in dict")
    logging.info("Done!")

def get_sentences(input_file):
    sentences = []
    words_set = set()
    with open(input_file, "r") as f:
        for line in tqdm(f, desc="get_sentences"):
            line = line.strip()
            words = line.split()
            if len(words) > 1:
                sentences.append(words)
                words_set |= set(words)
    logging.info("get sentences length:"+str(len(sentences)))
    return sentences, words_set


if __name__ == "__main__":
    sentences, words_set = get_sentences(config.input_file)
    model = train(sentences, window_size=5, iter=3)
    get_embeddings(model, words_set, config.output_file)
