import sys
sys.path.append("../..")

import torch
from torch.utils.data import Dataset
from .train_dataloader import process_context_item
from torch.utils.data import IterableDataset
from utils.args import config
import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)

locale_dict = {"DE": 1, "JP": 2, "UK": 3, "ES": 4, "FR": 5, "IT": 6}

class TestDataset(Dataset):
    def __init__(self, file, need_label=True, max_seq_len=128):
        self.max_seq_len = max_seq_len
        self.file = file
        self.need_label = need_label
        self.sid_list = []
        self.buffer = self.load_data()

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]

    def load_data(self):
        buffer = []
        logging.info('load test file ' + self.file)
        with open(self.file, "r") as fd:
            for line in fd.readlines():
                line_list = line.strip('\n').split('\t')
                if len(line_list) < 3:
                    continue

                id_list, candi_id, locale = line_list
                locale_code = locale_dict[locale]
                candi_id = int(candi_id)
                candi_id = torch.tensor([candi_id], dtype=torch.int32)
                locale_code = int(locale_code)
                locale_code = torch.tensor([locale_code], dtype=torch.int32)

                prev_ids = [int(item) for item in id_list.split(",")]

                prev_ids, padding_mask = process_context_item(prev_ids, self.max_seq_len)

                buffer.append([prev_ids, padding_mask, candi_id, locale_code])
        return buffer