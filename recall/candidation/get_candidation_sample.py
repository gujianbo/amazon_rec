import os
import sys
sys.path.append("../..")
from utils.args import config
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd
import logging
from itertools import combinations
import itertools
from collections import Counter
import hashlib


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)


def load_product(product_file):
    df = pd.read_csv(product_file)
    df_dic = df.groupby("id").locale.apply(set)
    print("load_product done!")
    return df_dic.to_dict()


def load_hot(root_path):
    import pickle
    hot_dict = pickle.load(open(f'{root_path}/hot_pre.dict', 'rb'))
    print("load_hot done!")
    return hot_dict


def load_recall(root_path):
    covi = "covisit_pre"
    sw = "swing_pre"
    import pickle
    co_global = pickle.load(open(f'{root_path}/{covi}/co.global.dict', 'rb'))
    print("load_recall global done!")
    co_de = pickle.load(open(f'{root_path}/{covi}/co.DE.dict', 'rb'))
    print("load_recall DE done!")
    co_jp = pickle.load(open(f'{root_path}/{covi}/co.JP.dict', 'rb'))
    print("load_recall JP done!")
    co_uk = pickle.load(open(f'{root_path}/{covi}/co.UK.dict', 'rb'))
    print("load_recall UK done!")
    co_it = pickle.load(open(f'{root_path}/{covi}/co.IT.dict', 'rb'))
    print("load_recall IT done!")
    co_es = pickle.load(open(f'{root_path}/{covi}/co.ES.dict', 'rb'))
    print("load_recall ES done!")
    co_fr = pickle.load(open(f'{root_path}/{covi}/co.FR.dict', 'rb'))
    print("load_recall FR done!")
    swing = pickle.load(open(f'{root_path}/{sw}/swing.sim', 'rb'))
    print("load_recall swing done!")

    recall_dict = dict()
    recall_dict["co_global"] = co_global
    recall_dict["co_DE"] = co_de
    recall_dict["co_JP"] = co_jp
    recall_dict["co_UK"] = co_uk
    recall_dict["co_IT"] = co_it
    recall_dict["co_ES"] = co_es
    recall_dict["co_FR"] = co_fr
    recall_dict["swing"] = swing
    return recall_dict


def get_candi(input_file, recall_dict, pro_dict, hot_dict, topk, single_topk, output_file, is_train=1):
    session_pd = pd.read_csv(input_file, sep=",")
    out_dict = {"prev_items": [], "next_item": [], "locale": [], "candi": []}
    for index, row in tqdm(session_pd.iterrows(), desc="gen_item_pair"):
        session = [sess.strip().strip("[").strip("]").strip("'") for sess in row["prev_items"].split()]
        prev_items = ",".join(session)
        hash_val = int(hashlib.md5((prev_items + str(config.seed)).encode('utf8')).hexdigest()[0:10], 16) % config.sample_cnt
        if hash_val != 0:
            continue
        if is_train == 1:
            next_item = row["next_item"]
        else:
            next_item = ""
        locale = row["locale"]
        unique_ids = list(dict.fromkeys(session[::-1]))

        ln = len(session)
        candidates = Counter()
        aids1 = list(itertools.chain(*[
            [rec_id for rec_id in recall_dict["swing"][aid] if rec_id in pro_dict and locale in pro_dict[rec_id]][:single_topk]
            for aid in session[::-1] if aid in recall_dict["swing"]]))
        for i, aid in enumerate(aids1):
            m = 0.1 + 0.9 * (ln - (i // 20)) / ln
            candidates[aid] += m

        local_dict = recall_dict["co_" + locale]
        aids3 = list(itertools.chain(*[
            [rec_id for rec_id in local_dict[aid] if
             rec_id in pro_dict and locale in pro_dict[rec_id]][:single_topk]
            for aid in session[::-1] if aid in local_dict]))
        for i, aid in enumerate(aids3):
            candidates[aid] += 0.5

        aids2 = list(itertools.chain(*[
            [rec_id for rec_id in recall_dict["co_global"][aid] if
             rec_id in pro_dict and locale in pro_dict[rec_id]][:single_topk]
            for aid in session[::-1] if aid in recall_dict["co_global"]]))
        for i, aid in enumerate(aids2):
            candidates[aid] += 0.5

        candi_len = len(candidates)
        top_candi = [k for k, v in candidates.most_common(candi_len) if k not in unique_ids]

        result = unique_ids + top_candi
        result = result + hot_dict[locale]
        out_dict["prev_items"].append(",".join(session))
        out_dict["next_item"].append(next_item)
        out_dict["locale"].append(locale)
        out_dict["candi"].append(",".join(result))

    out_pd = pd.DataFrame(out_dict)
    out_pd.to_csv(output_file)


if __name__ == "__main__":
    logging.info("input_file:" + config.input_file)
    logging.info("output_file:" + config.output_file)
    logging.info("product_file:" + config.product_file)
    logging.info("root_path:" + config.root_path)
    logging.info("topk:" + str(config.topk))
    logging.info("single_topk:" + str(config.single_topk))

    pro_dict = load_product(config.product_file)
    recall_dict = load_recall(config.root_path)
    hot_dict = load_hot(config.root_path)

    get_candi(config.input_file, recall_dict, pro_dict, hot_dict, config.topk, config.single_topk, config.output_file,
              config.is_train)