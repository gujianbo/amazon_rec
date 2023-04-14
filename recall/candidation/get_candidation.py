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

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)


def load_product(product_file):
    df = pd.read_csv(product_file)
    df_dic = df.groupby("id").locale.apply(set)
    return df_dic.to_dict()


def load_recall(root_path):
    import pickle
    co_global = pickle.load(open(f'{root_path}/covisit/co.global.dict', 'rb'))
    co_de = pickle.load(open(f'{root_path}/covisit/co.DE.dict', 'rb'))
    co_jp = pickle.load(open(f'{root_path}/covisit/co.JP.dict', 'rb'))
    co_uk = pickle.load(open(f'{root_path}/covisit/co.UK.dict', 'rb'))
    co_it = pickle.load(open(f'{root_path}/covisit/co.IT.dict', 'rb'))
    co_es = pickle.load(open(f'{root_path}/covisit/co.ES.dict', 'rb'))
    co_fr = pickle.load(open(f'{root_path}/covisit/co.FR.dict', 'rb'))
    swing = pickle.load(open(f'{root_path}/swing/swing.sim', 'rb'))

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


def get_candi(input_file, recall_dict, product_file):
    session_pd = pd.read_csv(input_file, sep=",")
    for index, row in tqdm(session_pd.iterrows(), desc="gen_item_pair"):
        session = [sess.strip().strip("[").strip("]").strip("'") for sess in row["prev_items"].split()]
        next_item = row["next_item"]
        locale = row["locale"]

        ln = len(session)
        candidates = Counter()
        aids1 = list(itertools.chain(*[
            [rec_id for rec_id in recall_dict["swing"][aid] if rec_id in product_file and locale in product_file[rec_id]][:20]
            for aid in session[::-1] if aid in recall_dict["swing"]]))
        for i, aid in enumerate(aids1):
            m = 0.1 + 0.9 * (ln - (i // 20)) / ln
            candidates[aid] += m

        aids2 = list(itertools.chain(*[
            [rec_id for rec_id in recall_dict["co_global"][aid] if
             rec_id in product_file and locale in product_file[rec_id]][:20]
            for aid in session[::-1] if aid in recall_dict["co_global"]]))
        for i, aid in enumerate(aids2):
            candidates[aid] += 1

        local_dict = recall_dict["co_" + locale]
        aids3 = list(itertools.chain(*[
            [rec_id for rec_id in local_dict[aid] if
             rec_id in product_file and locale in product_file[rec_id]][:20]
            for aid in session[::-1] if aid in local_dict]))
        for i, aid in enumerate(aids3):
            candidates[aid] += 1





if __name__ == "__main__":
    logging.info("input_file:" + config.input_file)
    logging.info("output_file:" + config.output_file)
    logging.info("product_file:" + config.product_file)
    logging.info("root_path:" + config.root_path)

    pro_dict = load_product(config.product_file)
    recall_dict = load_recall(config.root_path)

    get_candi(config.input_file, recall_dict)