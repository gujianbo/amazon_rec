import faiss
import numpy as np
from tqdm.auto import tqdm
import cmath
import pandas as pd
import sys
sys.path.append("../..")
from utils.args import config

import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)


def load_product(product_file):
    df = pd.read_csv(product_file)
    df_dic = df.groupby("id").locale.apply(set)
    print("load_product done!")
    return df_dic.to_dict()


def norm(feat_arr):
    sum = 0
    for i in feat_arr:
        sum += i*i
    return [item/cmath.sqrt(sum+0.0000001) for item in feat_arr]


def make_index(input_file, dim, product_dict, country, product2id, use_gpu, batch_size=128):
    index_ip = faiss.IndexFlatIP(dim)
    index_ip = faiss.IndexIDMap(index_ip)
    if use_gpu == 1:
        res = faiss.StandardGpuResources()
        index_ip = faiss.index_cpu_to_gpu(res, 0, index_ip)

    with open(input_file, "r") as f:
        ids = []
        features = []
        for line in tqdm(f, desc="make_index"):
            if len(ids) >= batch_size:
                ids_np = np.array(ids).astype('int64')
                features_np = np.array(features).astype('float32')
                index_ip.add_with_ids(features_np, ids_np)
                ids = []
                features = []
            line = line.strip()
            id, feat_str = line.split("\t")
            if id not in product_dict or country not in product_dict[id]:
                continue
            ids.append(int(product2id[id]))
            feat_arr = list(map(float, feat_str.split(",")))
            feat_arr_norm = norm(feat_arr)
            assert len(feat_arr) == dim
            features.append(feat_arr_norm)
        if len(ids) >= 0:
            ids_np = np.array(ids).astype('int64')
            features_np = np.array(features).astype('float32')
            index_ip.add_with_ids(features_np, ids_np)

    return index_ip


def search(input_file, output_file, dim, index_ip, product_dict, country, id2product, batch_size=128, topk=100):
    fdout = open(output_file, "w")
    with open(input_file, "r") as f:
        query_ids = []
        features = []
        for line in tqdm(f, desc="search"):
            if len(query_ids) >= batch_size:
                features_np = np.array(features).astype('float32')
                distances, ids = index_ip.search(features_np, topk+1)
                for i in range(len(query_ids)):
                    query_id = query_ids[i]
                    for j in range(len(ids[i])):
                        id = id2product[ids[i][j]]
                        score = distances[i][j]
                        if query_id != id:
                            fdout.write(str(query_id) + "\t" + str(id) + "\t" + str(score) + "\n")
                query_ids = []
                features = []
            line = line.strip()
            id, feat_str = line.split("\t")
            if id not in product_dict or country not in product_dict[id]:
                continue

            query_ids.append(id)
            feat_arr = list(map(float, feat_str.split(",")))
            assert len(feat_arr) == dim
            feat_arr_norm = norm(feat_arr)
            features.append(feat_arr_norm)
        if len(query_ids) >= 0:
            features_np = np.array(features).astype('float32')
            distances, ids = index_ip.search(features_np, topk)
            for i in range(len(query_ids)):
                query_id = query_ids[i]
                for j in range(len(ids[i])):
                    id = id2product[ids[i][j]]
                    score = distances[i][j]
                    if query_id != id:
                        fdout.write(str(query_id) + "\t" + str(id) + "\t" + str(score) + "\n")
    fdout.close()


if __name__ == "__main__":
    product_dict = load_product(config.product_file)
    import pickle
    product2id = pickle.load(open(config.product_dict_file, 'rb'))
    id2product = pickle.load(open(config.idproduct_dict_file, "rb"))
    index_ip = make_index(config.input_file, config.dim, product_dict, config.country, product2id, config.use_gpu, config.batch_size)
    search(config.input_file, config.output_file, config.dim, index_ip, product_dict, config.country, id2product, config.batch_size, config.topk)
