import sys
sys.path.append("../..")
import os
import torch
from torch.utils.data import IterableDataset
import random
from utils.args import config
import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)


def process_context_item(prev_ids, max_seq_len=128):
    prev_ids = prev_ids[::-1]

    padding_mask = [0] * len(prev_ids)
    if len(prev_ids) < max_seq_len:
        prev_ids += [config._PAD_] * (max_seq_len - len(prev_ids))
        padding_mask += [1] * (max_seq_len - len(padding_mask))
    else:
        prev_ids = prev_ids[:max_seq_len]
        padding_mask = padding_mask[:max_seq_len]
    padding_mask = torch.tensor(padding_mask, dtype=torch.int32)
    prev_ids = torch.tensor(prev_ids, dtype=torch.float32)

    return prev_ids, padding_mask


class TrainDatasetListBuffer(IterableDataset):
    def __init__(self, file_path, buffer_size=10000, need_label=True, max_seq_len=128):
        self.max_seq_len = max_seq_len
        self.file_path = file_path
        self.files = os.listdir(self.file_path)
        self.buffer_size = buffer_size
        self.need_label = need_label
        self.ignore_just_label = False
        random.shuffle(self.files)
        self.files = [self.file_path+"/"+f for f in self.files]

    def __iter__(self):
        buffer = []
        for file in self.files:
            logging.info('load file ' + file)
            line_num = 0
            with open(file, "r") as fd:
                for line in fd:
                    line_num += 1
                    if line_num % 100000 == 0:
                        logging.info(f"file {file}: {line_num} lines scaned!")
                    line_list = line.strip('\n').split('\t')
                    if len(line_list) < 8:
                        continue
                    prev_ids, candi, candi_id, locale_code, item_feat_str, session_stat_feat_str, interact_feat_str, label = line_list
                    candi_id = int(candi_id)
                    locale_code = int(locale_code)
                    item_feat = [float(item) for item in item_feat_str.split(",")]
                    session_stat_feat = [float(item) for item in session_stat_feat_str.split(",")]
                    interact_feat = [float(item) for item in interact_feat_str.split(",")]
                    dense_feat = []
                    dense_feat += item_feat
                    dense_feat += session_stat_feat
                    dense_feat += interact_feat

                    prev_ids = [int(item) for item in prev_ids.split(",")]
                    label = float(label)

                    prev_ids, padding_mask = process_context_item(prev_ids, self.max_seq_len)

                    if len(buffer) >= self.buffer_size:
                        idx = random.randint(0, self.buffer_size - 1)
                        # logging.info(f"item:{buffer[idx]}")
                        yield buffer[idx]
                        buffer[idx] = [prev_ids, padding_mask, locale_code, dense_feat, candi_id, label]
                    else:
                        buffer.append([prev_ids, padding_mask, locale_code, dense_feat, candi_id, label])
                while len(buffer) > 0:
                    yield buffer.pop()