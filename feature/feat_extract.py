
import os
import sys
sys.path.append("..")
from utils.args import config
from tqdm.auto import tqdm
import pandas as pd
import csv


def load_item_feat(item_feat_file):
    import pickle
    item_dict = pickle.load(open(item_feat_file, 'rb'))
    print("load_item_feat done!")
    return item_dict


def load_i2i_dicts(path):
    co_global = load_i2i_dict(path+"/covisit/pairs.global.csv")
    print("load_i2i_dict global done!")
    co_DE = load_i2i_dict(path+"/covisit/pairs.DE.csv")
    print("load_i2i_dict DE done!")
    co_JP = load_i2i_dict(path+"/covisit/pairs.JP.csv")
    print("load_i2i_dict JP done!")
    co_UK = load_i2i_dict(path+"/covisit/pairs.UK.csv")
    print("load_i2i_dict UK done!")
    co_ES = load_i2i_dict(path+"/covisit/pairs.ES.csv")
    print("load_i2i_dict ES done!")
    co_FR = load_i2i_dict(path+"/covisit/pairs.FR.csv")
    print("load_i2i_dict FR done!")
    co_IT = load_i2i_dict(path+"/covisit/pairs.IT.csv")
    print("load_i2i_dict IT done!")
    swing = load_i2i_dict(path+"/swing/swing.sim.full", sep="\t")
    print("load_i2i_dict swing done!")
    return {"co_global": co_global, "co_DE": co_DE, "co_JP": co_JP, "co_UK": co_UK,
            "co_ES": co_ES, "co_FR": co_FR, "co_IT": co_IT, "swing": swing}


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

    last_socre = []
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
        last_socre += [co_score, co_local_score, swing_score]
    if len(last_socre) < 60:
        last_socre += [0]*(60-len(last_socre))
    return [in_session, re_cnt, last_idx_span, max_local_co_score, max_co_score, max_swing_score,
            sum_local_co_score/len(prev_items), sum_co_score/len(prev_items), sum_swing_score/len(prev_items)] + last_socre


def feat_extract(input_file, output_file, item_dict, i2i_dicts):
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
        interact_feat_str = ','.join([str(item) for item in interact_feat])

        fdout.write(f"{prev_items}\t{candi}\t{locale_code}\t{item_feat_str}\t{session_stat_feat_str}\t{interact_feat_str}\t{label}\n")
    fdout.close()
    fd.close()


if __name__ == "__main__":
    print(f"input_file:{config.input_file}")
    print(f"output_file:{config.output_file}")
    print(f"item_feat_file:{config.item_feat_file}")
    print(f"root_path:{config.root_path}")
    item_dict = load_item_feat(config.item_feat_file)
    i2i_dicts = load_i2i_dicts(config.root_path)
    feat_extract(config.input_file, config.output_file, item_dict, i2i_dicts)
