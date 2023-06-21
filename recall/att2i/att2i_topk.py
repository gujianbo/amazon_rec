import xgboost
import os
import sys
sys.path.append("../..")
from utils.args import config
from tqdm.auto import tqdm
import hashlib
import logging
import numpy as np
import pandas as pd
import faiss

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)


def map_id(input_file, vec_file, idproduct_dict_file, output_file):
    df = pd.read_csv(input_file)
    prod_dic = df.groupby("id").locale.apply(set)
    import pickle
    id2prod = pickle.load(open(idproduct_dict_file, 'rb'))
    fd = open(vec_file, "r")
    fdout = dict()
    fdout["DE"] = open(output_file+"_DE", "w")
    fdout["JP"] = open(output_file+"_JP", "w")
    fdout["UK"] = open(output_file+"_UK", "w")
    fdout["IT"] = open(output_file+"_IT", "w")
    fdout["FR"] = open(output_file+"_FR", "w")
    fdout["ES"] = open(output_file+"_ES", "w")
    for line in fd:
        item_id, vec = line.strip().split("\t")
        item_id = int(item_id)
        prod_id = id2prod[item_id]
        cnty_set = prod_dic[prod_id]
        for cnty in cnty_set:
            fdout[cnty].write(line)
    for key in fdout:
        fdout[key].close()


def make_index(input_file, dim, use_gpu, batch_size=128):
    index_ip = faiss.IndexFlatIP(dim)
    index_ip = faiss.IndexIDMap(index_ip)
    if use_gpu == 1:
        res = faiss.StandardGpuResources()
        index_ip = faiss.index_cpu_to_gpu(res, 0, index_ip)

    with open(input_file, "r") as f:
        ids = []
        features = []
        for line in tqdm(f, desc="make_index "+input_file):
            if len(ids) >= batch_size:
                ids_np = np.array(ids).astype('int64')
                features_np = np.array(features).astype('float32')
                index_ip.add_with_ids(features_np, ids_np)
                ids = []
                features = []
            line = line.strip()
            id, feat_str = line.split("\t")
            ids.append(int(id))
            feat_arr = list(map(float, feat_str.split(",")))
            # feat_arr_norm = norm(feat_arr)
            assert len(feat_arr) == dim
            features.append(feat_arr)
        if len(ids) >= 0:
            ids_np = np.array(ids).astype('int64')
            features_np = np.array(features).astype('float32')
            index_ip.add_with_ids(features_np, ids_np)

    return index_ip


locale_dict = {1: "DE", 2: "JP", 3: "UK", 4: "ES", 5: "FR", 6: "IT"}


def get_topk(input_file, item_file, output_file, dim=128, topk=100, batch_size=128, use_gpu=1):
    fd = open(input_file, "r")
    fdout = open(output_file, "w")
    index_ip = make_index(item_file + "_DE", dim, use_gpu, batch_size)
    next_item_ids = []
    features = []
    countrys = []
    last_country_code = 1
    for line in tqdm(fd, desc="get_topk "+input_file):
        if len(next_item_ids) >= batch_size:
            features_np = np.array(features).astype('float32')
            distances, ids = index_ip.search(features_np, topk)
            for i in range(len(next_item_ids)):
                next_item_id = next_item_ids[i]
                cand_ids = []
                for j in range(len(ids[i])):
                    id = ids[i][j]
                    score = distances[i][j]
                    cand_ids.append(id)
                cand_ids_str = ",".join([str(it) for it in cand_ids])
                fdout.write(f"{countrys[i]}\t{next_item_id}\t{cand_ids_str}\n")
            countrys = []
            features = []
            next_item_ids = []
        user_vec, country_code, next_item = line.strip().split("\t")
        country_code = int(country_code)
        if last_country_code != country_code:
            if len(next_item_ids) > 0:
                features_np = np.array(features).astype('float32')
                distances, ids = index_ip.search(features_np, topk)
                for i in range(len(next_item_ids)):
                    next_item_id = next_item_ids[i]
                    cand_ids = []
                    for j in range(len(ids[i])):
                        id = ids[i][j]
                        score = distances[i][j]
                        cand_ids.append(id)
                    cand_ids_str = ",".join([str(it) for it in cand_ids])
                    fdout.write(f"{countrys[i]}\t{next_item_id}\t{cand_ids_str}\n")
                countrys = []
                features = []
                next_item_ids = []
            cnty = locale_dict[country_code]
            print("load dict make_index:" + item_file + "_" + cnty)
            index_ip = make_index(item_file + "_" + cnty, dim, use_gpu, batch_size)

        next_item_ids.append(next_item)
        countrys.append(country_code)
        feat_arr = list(map(float, user_vec.split(",")))
        assert len(feat_arr) == dim
        features.append(feat_arr)
        last_country_code = country_code

    if len(next_item_ids) >= 0:
        features_np = np.array(features).astype('float32')
        distances, ids = index_ip.search(features_np, topk + 1)
        for i in range(len(next_item_ids)):
            next_item_id = next_item_ids[i]
            cand_ids = []
            for j in range(len(ids[i])):
                id = ids[i][j]
                score = distances[i][j]
                cand_ids.append(id)
            cand_ids_str = ",".join([str(it) for it in cand_ids])
            fdout.write(f"{countrys[i]}\t{next_item_id}\t{cand_ids_str}\n")


if __name__ == "__main__":
    # map_id(config.product_file, config.itemvec_file, config.idproduct_dict_file, config.item_file)
    get_topk(config.input_file, config.item_file, config.output_file, config.dim, config.topk, config.batch_size, config.use_gpu)