
import os
import sys
sys.path.append("..")
from utils.args import config
from tqdm.auto import tqdm
import pandas as pd
import csv
import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)

def load_item_feat(item_feat_file):
    import pickle
    item_dict = pickle.load(open(item_feat_file, 'rb'))
    logging.info("load_item_feat done!")
    return item_dict


def load_i2i_dicts(path):
    covi = "covisit_pre"
    sw = "swing_pre"
    sw2 = "swing"
    n2v = "n2v"
    co_global = load_i2i_dict(path+f"/{covi}/pairs.global.csv")
    logging.info("load_i2i_dict global done!")
    co_DE = load_i2i_dict(path+f"/{covi}/pairs.DE.csv")
    logging.info("load_i2i_dict DE done!")
    co_JP = load_i2i_dict(path+f"/{covi}/pairs.JP.csv")
    logging.info("load_i2i_dict JP done!")
    co_UK = load_i2i_dict(path+f"/{covi}/pairs.UK.csv")
    logging.info("load_i2i_dict UK done!")
    co_ES = load_i2i_dict(path+f"/{covi}/pairs.ES.csv")
    logging.info("load_i2i_dict ES done!")
    co_FR = load_i2i_dict(path+f"/{covi}/pairs.FR.csv")
    logging.info("load_i2i_dict FR done!")
    co_IT = load_i2i_dict(path+f"/{covi}/pairs.IT.csv")
    logging.info("load_i2i_dict IT done!")

    swing = load_i2i_dict(path+f"/{sw}/swing.sim.full", sep="\t")
    logging.info("load_i2i_dict swing done!")
    swing_DE = load_i2i_dict(path+f"/{sw2}_DE/swing.sim.full", sep="\t")
    logging.info("load_i2i_dict swing_DE done!")
    swing_UK = load_i2i_dict(path+f"/{sw2}_UK/swing.sim.full", sep="\t")
    logging.info("load_i2i_dict swing_UK done!")
    swing_JP = load_i2i_dict(path+f"/{sw2}_JP/swing.sim.full", sep="\t")
    logging.info("load_i2i_dict swing_JP done!")
    swing_ES = load_i2i_dict(path+f"/{sw2}_ES/swing.sim.full", sep="\t")
    logging.info("load_i2i_dict swing_ES done!")
    swing_FR = load_i2i_dict(path+f"/{sw2}_FR/swing.sim.full", sep="\t")
    logging.info("load_i2i_dict swing_FR done!")
    swing_IT = load_i2i_dict(path+f"/{sw2}_IT/swing.sim.full", sep="\t")
    logging.info("load_i2i_dict swing_IT done!")

    n2v_de = load_i2i_dict(path+f"/{n2v}/i2i_DE.txt", sep="\t")
    logging.info("load_i2i_dict n2v_de done!")
    n2v_jp = load_i2i_dict(path+f"/{n2v}/i2i_JP.txt", sep="\t")
    logging.info("load_i2i_dict n2v_jp done!")
    n2v_uk = load_i2i_dict(path+f"/{n2v}/i2i_UK.txt", sep="\t")
    logging.info("load_i2i_dict n2v_uk done!")
    n2v_es = load_i2i_dict(path+f"/{n2v}/i2i_ES.txt", sep="\t")
    logging.info("load_i2i_dict n2v_es done!")
    n2v_fr = load_i2i_dict(path+f"/{n2v}/i2i_FR.txt", sep="\t")
    logging.info("load_i2i_dict n2v_fr done!")
    n2v_it = load_i2i_dict(path+f"/{n2v}/i2i_IT.txt", sep="\t")
    logging.info("load_i2i_dict n2v_it done!")
    return {"co_global": co_global, "co_DE": co_DE, "co_JP": co_JP, "co_UK": co_UK, "co_ES": co_ES, "co_FR": co_FR, "co_IT": co_IT,
            "swing": swing, "swing_DE": swing_DE, "swing_JP": swing_JP, "swing_UK": swing_UK, "swing_ES": swing_ES, "swing_FR": swing_FR, "swing_IT": swing_IT,
            "n2v_DE": n2v_de, "n2v_JP": n2v_jp, "n2v_UK": n2v_uk, "n2v_ES": n2v_es, "n2v_FR": n2v_fr, "n2v_IT": n2v_it
            }


def load_i2i_dict(file, sep=","):
    fd = open(file, "r")
    i2i_dict = dict()
    for line in fd:
        item1, item2, score = line.split(sep)
        if item1 == "item1":
            continue
        score = float(score)
        key = item1 + "," + item2
        if item1 > item2:
            key = item2 + "," + item1
        i2i_dict[key] = score
    return i2i_dict


def session_stat(prev_items):
    session_len = len(prev_items)
    if session_len == 0:
        return [0]*3
    session_set = set(prev_items)
    session_uniq_len = len(session_set)
    re_visit_cnt = session_len/session_uniq_len
    return [session_len, session_uniq_len, re_visit_cnt]


def interact_stat(prev_items, candi, i2i_dicts, locale):
    if len(prev_items) == 0:
        return [0] * 69
    session_cnt = dict()
    in_session = 0
    in_session_last_idx = -1

    max_local_co_score = 0
    max_co_score = 0
    max_swing_score = 0

    sum_local_co_score = 0
    sum_co_score = 0
    sum_swing_score = 0

    for i in range(len(prev_items)):
        item = prev_items[i]
        if item == candi:
            in_session = 1
            in_session_last_idx = i

        key = item+","+candi
        if item > candi:
            key = candi+","+item
        if key in i2i_dicts["co_global"]:
            tmp_score = i2i_dicts["co_global"][key]
            if tmp_score > max_co_score:
                max_co_score = tmp_score
            sum_co_score += tmp_score
        if key in i2i_dicts["co_"+locale]:
            tmp_score = i2i_dicts["co_"+locale][key]
            if tmp_score > max_local_co_score:
                max_local_co_score = tmp_score
            sum_local_co_score += tmp_score
        if key in i2i_dicts["swing"]:
            tmp_score = i2i_dicts["swing"][key]
            if tmp_score > max_swing_score:
                max_swing_score = tmp_score
            sum_swing_score += tmp_score

        if item in session_cnt:
            session_cnt[item] += 1
        else:
            session_cnt[item] = 1

    if candi in session_cnt:
        last_idx_span = len(prev_items) - in_session_last_idx
        re_cnt = session_cnt[candi]
    else:
        last_idx_span = -1
        re_cnt = 0

    last_score = []
    for item in prev_items[::-1][:20]:
        key = item + "," + candi
        if item > candi:
            key = candi + "," + item
        if key in i2i_dicts["co_global"]:
            co_score = i2i_dicts["co_global"][key]
        else:
            co_score = 0
        if key in i2i_dicts["co_" + locale]:
            co_local_score = i2i_dicts["co_" + locale][key]
        else:
            co_local_score = 0
        if key in i2i_dicts["swing"]:
            swing_score = i2i_dicts["swing"][key]
        else:
            swing_score = 0
        if key in i2i_dicts["swing_" + locale]:
            swing_local_score = i2i_dicts["swing_" + locale][key]
        else:
            swing_local_score = 0

        if key in i2i_dicts["n2v_" + locale]:
            n2v_score = i2i_dicts["n2v_" + locale][key]
        else:
            n2v_score = 0

        last_score += [co_score, co_local_score, swing_score, swing_local_score, n2v_score]
    if len(last_score) < 100:
        last_score += [0]*(100-len(last_score))
    return [in_session, re_cnt, last_idx_span, max_local_co_score, max_co_score, max_swing_score,
            sum_local_co_score/len(prev_items), sum_co_score/len(prev_items), sum_swing_score/len(prev_items)] + last_score


def feat_extract(input_file, output_file, item_dict, i2i_dicts):
    logging.info("-----start feat_extract-----")
    locale_dict = {"DE": 1, "JP": 2, "UK": 3, "ES": 4, "FR": 5, "IT": 6}
    fdout = open(output_file, "w")

    fd = open(input_file, "r")
    csv_reader = csv.DictReader(fd, skipinitialspace=True)
    for row in tqdm(csv_reader, desc="feat_extract"):
        prev_items = row["prev_items"].split(",")
        locale = row["locale"]
        locale_code = locale_dict[locale]
        candi = row["recall_candi"]
        label = int(row["label"])
        if candi in item_dict:
            item_feat = item_dict[candi]
        else:
            item_feat = [0]*16
        item_feat_str = ','.join([str(item) for item in item_feat])

        session_stat_feat = session_stat(prev_items)
        session_stat_feat_str = ','.join([str(item) for item in session_stat_feat])

        interact_feat = interact_stat(prev_items, candi, i2i_dicts, locale)
        # interact_feat_str = ','.join([str(item) for item in interact_feat])

        local_idx = [3, 6,
                     10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 35, 37, 40, 42, 45, 47, 50, 52, 55, 57,
                     60, 62, 65, 67, 70, 72, 75, 77, 80, 82, 85, 87, 90, 92, 95, 97, 100, 102, 105, 107]

        interact_feat_glo = [interact_feat[idx] for idx in range(len(interact_feat)) if idx not in local_idx]
        interact_feat_glo_str = ','.join([str(item) for item in interact_feat_glo])
        local_feat = [interact_feat[idx] for idx in local_idx]
        empty_feat = [0.0] * 42
        local_sec_feat = []
        for i in range(locale_code - 1):
            local_sec_feat += empty_feat
        local_sec_feat += local_feat
        for i in range(6 - locale_code):
            local_sec_feat += empty_feat
        local_sec_feat_str = ','.join([str(item) for item in local_sec_feat])
        locale_code_feat = []
        for i in range(locale_code - 1):
            locale_code_feat += [0.0]
        locale_code_feat += [1.0]
        for i in range(6 - locale_code):
            locale_code_feat += [0.0]
        locale_code_feat_str = ','.join([str(item) for item in locale_code_feat])

        fdout.write(f"{row['prev_items']}\t{candi}\t{locale_code}\t{item_feat_str}\t{session_stat_feat_str}\t{interact_feat_glo_str}"
                    f"\t{local_sec_feat_str}\t{locale_code_feat_str}\t{label}\n")
    fdout.close()
    fd.close()


if __name__ == "__main__":
    logging.info(f"input_file:{config.input_file}")
    logging.info(f"output_file:{config.output_file}")
    logging.info(f"item_feat_file:{config.item_feat_file}")
    logging.info(f"root_path:{config.root_path}")
    item_dict = load_item_feat(config.item_feat_file)
    i2i_dicts = load_i2i_dicts(config.root_path)
    feat_extract(config.input_file, config.output_file, item_dict, i2i_dicts)
