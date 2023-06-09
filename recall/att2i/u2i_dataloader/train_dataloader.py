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
    prev_ids = torch.tensor(prev_ids, dtype=torch.int32)

    return prev_ids, padding_mask


locale_dict = {"DE": 1, "JP": 2, "UK": 3, "ES": 4, "FR": 5, "IT": 6}


class TrainDatasetListBuffer(IterableDataset):
    def __init__(self, file_path, buffer_size=10000, need_label=True, max_seq_len=128):
        self.max_seq_len = max_seq_len
        self.file_path = file_path
        self.files = os.listdir(self.file_path)
        self.buffer_size = buffer_size
        self.need_label = need_label
        self.ignore_just_label = False

    def __iter__(self):
        buffer = []
        logging.info('load file ' + self.file_path)
        line_num = 0
        with open(self.file_path, "r") as fd:
            for line in fd:
                line_num += 1
                if line_num % 100000 == 0:
                    logging.info(f"file {self.file_path}: {line_num} lines scaned!")
                line_list = line.strip('\n').split('\t')
                if len(line_list) < 3:
                    continue
                id_list, candi, locale = line_list
                locale_code = locale_dict[locale]
                candi_id = int(candi_id)
                candi_id = torch.tensor([candi_id], dtype=torch.int32)
                locale_code = int(locale_code)
                locale_code = torch.tensor([locale_code], dtype=torch.int32)

                prev_ids = [int(item) for item in id_list.split(",")]

                prev_ids, padding_mask = process_context_item(prev_ids, self.max_seq_len)

                if len(buffer) >= self.buffer_size:
                    idx = random.randint(0, self.buffer_size - 1)
                    # logging.info(f"item:{buffer[idx]}")
                    yield buffer[idx]
                    buffer[idx] = [prev_ids, padding_mask, candi_id, locale_code]
                else:
                    buffer.append([prev_ids, padding_mask, candi_id, locale_code])
            while len(buffer) > 0:
                yield buffer.pop()
