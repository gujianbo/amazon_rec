
import torch
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import numpy as np
import random
import sys
sys.path.append("..")
from dnn_model import DNNModel
from utils.args import config
from eval.eval_rank_mrr import test_mrr
from dataloader.train_dataloader import TrainDatasetListBuffer
from dataloader.test_dataloader import TestDatasetList
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import time

import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)

random.seed(config.seed+1)
random.seed(config.seed)
np.random.seed(config.seed)
print(config)

input_size = (config.max_seq_len + 2) * config.d_model + config.dense_size
model = DNNModel(
    input_size=input_size,
    num_items=config.num_items,
    d_model=config.d_model,
    num_layers=config.num_layers,
    num_head=config.num_head,
    d_ff=config.d_ff,
    max_len=config.max_seq_len,
    dropout=config.dropout
)
logging.info(f"input_size:{input_size} | num_items:{config.num_items} | d_model:{config.d_model} | "
             f"num_layers:{config.num_layers} | num_head:{config.num_head} | d_ff:{config.d_ff} | "
             f"max_len:{config.max_seq_len} | dropout:{config.dropout}")

if config.init_parameters != "":
    # print('load warm up model ', config.init_parameters, file=config.log_file)
    logging.debug('load warm up model '+config.init_parameters)
    ptm = torch.load(config.init_parameters)
    for k, v in model.state_dict().items():
        if not k in ptm:
            pass
            # print("warning: not loading " + k, file=config.log_file)
            logging.debug("warning: not loading " + k)
        else:
            # print("loading " + k, file=config.log_file)
            logging.debug("loading " + k)
            v.set_value(ptm[k])

optimizer = torch.optim.AdamW(
    lr=config.lr,
    params=model.parameters(),
    weight_decay=config.weight_decay
)
scheduler = StepLR(optimizer=optimizer, step_size=30, gamma=0.1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"device:{device}")
train_dataset = TrainDatasetListBuffer(config.train_file, buffer_size=config.buffer_size, need_label=True, max_seq_len=config.max_seq_len)
train_dataloader = DataLoader(train_dataset, config.train_batch_size)

test_dataset = TestDatasetList(config.test_file, need_label=True, max_seq_len=config.max_seq_len)
test_dataloader = DataLoader(test_dataset, config.test_batch_size)

idx = 0
bce_loss = nn.BCELoss()
version = int(time.time())
logging.info(f"version:{version}")
model.to(device)
logging.info(f"load model to {device} done!")


max_mrr = 0
for round in range(config.epoch):
    for train_data_batch in train_dataloader:
        model.train()
        # optimizer.clear_grad()
        optimizer.zero_grad()
        prev_ids, padding_mask, locale_code, dense_feat, candi_id, label = train_data_batch
        prev_ids, padding_mask, locale_code, dense_feat, candi_id = \
            prev_ids.to(device), padding_mask.to(device), locale_code.to(device), dense_feat.to(device), candi_id.to(device)

        if config.log_level >= 1:
            logging.debug(f"prev_ids.shape:{prev_ids.shape}, padding_mask.shape:{padding_mask.shape}, "
                          f"locale_code.shape:{locale_code.shape}, dense_feat.shape:{dense_feat.shape}, "
                          f"candi_id.shape:{candi_id.shape}, label.shape:{label.shape}")

        label_t = torch.tensor(label, dtype=torch.float32).to(device)
        logits = model(prev_ids, padding_mask, locale_code, candi_id, dense_feat)

        label_loss = bce_loss(logits, label_t.unsqueeze(1))
        label_loss.backward()
        optimizer.step()
        if idx % config.log_interval == 0:
            logging.info(f'{idx:5d}th step | label_loss {label_loss.cpu().item():5.6f}')
        if idx % config.eval_step == 0:
            model.eval()
            test_logits, test_labels = [], []
            eval_idx = 0
            test_label_loss_sum = 0
            for test_data_batch in test_dataloader:
                test_prev_ids, test_padding_mask, test_locale_code, test_dense_feat, test_candi_id, test_label = train_data_batch
                test_prev_ids, test_padding_mask, test_locale_code, test_dense_feat, test_candi_id = \
                    test_prev_ids.to(device), test_padding_mask.to(device), test_locale_code.to(device), test_dense_feat.to(
                        device), test_candi_id.to(device)
                test_label_t = torch.tensor(test_label, dtype=torch.float32).to(device)
                logits_test = model(test_prev_ids, test_padding_mask, test_locale_code, test_candi_id, test_dense_feat)
                test_logits += list(logits_test.detach().squeeze().cpu().numpy())
                test_labels += list(test_label_t.detach().numpy())

                test_label_loss = bce_loss(logits_test, test_label_t.unsqueeze(1).to(device))
                test_label_loss_sum += test_label_loss
                eval_idx += 1
            test_auc = roc_auc_score(test_labels, test_logits)
            test_loss = test_label_loss_sum/eval_idx
            test_mrr = test_mrr(test_dataloader.sid_list, test_logits, test_labels)
            logging.info(f'{idx:5d}th step | test_auc {test_auc:5.6f} | test_loss {test_loss:5.6f} | test_mrr {test_mrr:5.6f}')
        if idx >= config.eval_step and (test_mrr > max_mrr or idx % config.save_step == 0):
            if test_mrr > max_mrr:
                max_mrr = test_mrr
            model_name = f"{config.save_path}/dnn_v{version}_steps_{idx}_{test_mrr:.4f}_{test_auc:.4f}_{test_loss:.4f}.model"
            torch.save(model.state_dict(), model_name)
            logging.info(f"model {model_name} is saved!")
        idx += 1
    scheduler.step()

model_name = f"{config.save_path}/dnn_v{version}_steps_{idx}_{test_mrr:.4f}_{test_auc:.4f}_{test_loss:.4f}.model"
torch.save(model.state_dict(), model_name)
logging.info(f"model {model_name} is saved!")
