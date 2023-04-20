import os
import sys
sys.path.append("..")
from utils.args import config
from tqdm.auto import tqdm
import pandas as pd


def get_seq(item):
    session = [sess.strip().strip("[").strip("]").strip("'") for sess in item.prev_items.split()]
    session.append(item.next_item)
    return session


def stat(item):
    sum = len(item)  # 总互动量
    clk_sum = len(item["index"].unique())  # 不同session的互动量
    clk_rate = clk_sum/3606249  # 互动率
    avg_clk_sum = sum/clk_sum  # 平均每个session互动量
    de_clk_sum = len(item.loc[item.locale == 'DE']["index"].unique())
    de_clk_rate = de_clk_sum/1111416
    jp_clk_sum = len(item.loc[item.locale == 'JP']["index"].unique())
    jp_clk_rate = jp_clk_sum/979119
    uk_clk_sum = len(item.loc[item.locale == 'UK']["index"].unique())
    uk_clk_rate = uk_clk_sum/1182181
    es_clk_sum = len(item.loc[item.locale == 'ES']["index"].unique())
    es_clk_rate = es_clk_sum/89047
    fr_clk_sum = len(item.loc[item.locale == 'FR']["index"].unique())
    fr_clk_rate = fr_clk_sum/117561
    it_clk_sum = len(item.loc[item.locale == 'IT']["index"].unique())
    it_clk_rate = it_clk_sum/126925
    return (sum, avg_clk_sum, clk_sum, clk_rate, de_clk_sum, de_clk_rate, jp_clk_sum, jp_clk_rate,
            uk_clk_sum, uk_clk_rate, es_clk_sum, es_clk_rate, fr_clk_sum, fr_clk_rate, it_clk_sum, it_clk_rate)


def item_feature_stat(input_file, output_file):
    session_pd = pd.read_csv(input_file, sep=",")
    session_pd["prev_items"] = session_pd.apply(get_seq, axis=1)
    session_pd = session_pd.explode("prev_items").reset_index()
    session_pd = session_pd.groupby("prev_items").apply(stat)
    item_feat = session_pd.to_dict()
    import pickle
    with open(output_file, 'wb') as fd:
        pickle.dump(item_feat, fd)


if __name__ == "__main__":
    print(f"input_file:{config.input_file}")
    print(f"output_file:{config.output_file}")
    item_feature_stat(config.input_file, config.output_file)
