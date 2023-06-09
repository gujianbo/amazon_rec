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


class TestDatasetList(Dataset):
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
                if len(line_list) < 8:
                    continue

                prev_ids, candi, candi_id, locale_code, item_feat_str, session_stat_feat_str, interact_feat_str, label = line_list
                self.sid_list.append(prev_ids)

                candi_id = int(candi_id)
                candi_id = torch.tensor([candi_id], dtype=torch.int32)
                locale_code = int(locale_code)
                locale_code = torch.tensor([locale_code], dtype=torch.int32)
                item_feat = [float(item) for item in item_feat_str.split(",")]
                session_stat_feat = [float(item) for item in session_stat_feat_str.split(",")]
                interact_feat = [float(item) for item in interact_feat_str.split(",")]
                dense_feat = []
                dense_feat += item_feat
                dense_feat += session_stat_feat
                dense_feat += interact_feat

                local_idx = [3, 6, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67]
                local_feat = [interact_feat[idx] for idx in local_idx]
                empty_feat = [0.0] * 22
                local_sec_feat = []
                for i in range(locale_code - 1):
                    local_sec_feat += empty_feat
                local_sec_feat += local_feat
                for i in range(6 - locale_code):
                    local_sec_feat += empty_feat

                dense_feat += local_sec_feat
                dense_feat = torch.tensor(dense_feat, dtype=torch.float32)

                prev_ids = [int(item) for item in prev_ids.split(",")]
                label = float(label)

                prev_ids, padding_mask = process_context_item(prev_ids, self.max_seq_len)
                buffer.append([prev_ids, padding_mask, locale_code, dense_feat, candi_id, label])
        return buffer


class SubmissionDatasetList(IterableDataset):
    def __init__(self, file, need_label=True, max_seq_len=128, local_code=[1, 2, 3]):
        self.max_seq_len = max_seq_len
        self.file = file
        self.need_label = need_label
        self.local_code = local_code

    def __iter__(self):
        logging.info('load submission file ' + self.file)
        with open(self.file, "r") as fd:
            for line in fd.readlines():
                line_list = line.strip('\n').split('\t')
                if len(line_list) < 8:
                    continue

                prev_ids, candi, candi_id, locale_code, item_feat_str, session_stat_feat_str, interact_feat_str, label = line_list
                locale_code = int(locale_code)
                if locale_code not in self.local_code:
                    continue

                candi_id = int(candi_id)
                candi_id = torch.tensor([candi_id], dtype=torch.int32)
                locale_code = torch.tensor([locale_code], dtype=torch.int32)
                item_feat = [float(item) for item in item_feat_str.split(",")]
                session_stat_feat = [float(item) for item in session_stat_feat_str.split(",")]
                interact_feat = [float(item) for item in interact_feat_str.split(",")]
                dense_feat = []
                dense_feat += item_feat
                dense_feat += session_stat_feat
                dense_feat += interact_feat

                local_idx = [3, 6, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67]
                local_feat = [interact_feat[idx] for idx in local_idx]
                empty_feat = [0.0] * 22
                local_sec_feat = []
                for i in range(locale_code - 1):
                    local_sec_feat += empty_feat
                local_sec_feat += local_feat
                for i in range(6 - locale_code):
                    local_sec_feat += empty_feat

                dense_feat += local_sec_feat
                dense_feat = torch.tensor(dense_feat, dtype=torch.float32)

                prev_ids = [int(item) for item in prev_ids.split(",")]
                label = float(label)

                prev_ids, padding_mask = process_context_item(prev_ids, self.max_seq_len)
                yield [prev_ids, padding_mask, locale_code, dense_feat, candi_id, label]
