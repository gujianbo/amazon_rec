import sys
sys.path.append("../..")
import torch
from torch.optim.lr_scheduler import StepLR
import numpy as np
import random
from att2i_model import AttentionU2I
from utils.args import config
from u2i_dataloader.train_dataloader import TrainDataset
from u2i_dataloader.test_dataloader import TestDataset
from torch.utils.data import DataLoader
import time

import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)

random.seed(config.seed)
np.random.seed(config.seed)
print(config)

input_size = (config.max_seq_len + 2) * config.d_model
model = AttentionU2I(
    num_items=config.num_items,
    num_country=7,
    num_layers=config.num_layers,
    num_head=config.num_head,
    d_model=config.d_model,
    d_ff=config.d_ff,
    max_len=config.max_seq_len,
    dropout=config.dropout,
    input_size=input_size,
    hidden_size=config.u2i_hidden_size,
    emb_size=config.u2i_emb_size
    # top_cross=config.top_cross,
    # prerank_logits=config.prerank_logits
)

if config.init_parameters != "":
    # print('load warm up model ', config.init_parameters, file=config.log_file)
    logging.debug('load warm up model '+config.init_parameters)
    model_dict = model.state_dict()
    ptm = torch.load(config.init_parameters)
    state_dict = {k: v for k, v in ptm.items() if k in model_dict.keys()}

    for k, v in state_dict.items():
        logging.debug(f"loading {k}, shape:{v.shape}")
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

decay_params = [
    p.name for n, p in model.named_parameters()
    if not any(nd in n for nd in ["bias", "norm"])
]

optimizer = torch.optim.AdamW(
    lr=config.lr,
    params=model.parameters(),
    weight_decay=config.weight_decay
)
scheduler = StepLR(optimizer=optimizer, step_size=3, gamma=0.1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"device:{device}")

train_dataset = TrainDataset(file_path=config.train_file, max_seq_len=config.max_seq_len)
train_dataloader = DataLoader(train_dataset, config.train_batch_size)
test_dataset = TestDataset(file=config.test_file, max_seq_len=config.max_seq_len)
test_dataloader = DataLoader(test_dataset, config.test_batch_size)

version = int(time.time())
logging.info(f"version:{version}")
model.to(device)
logging.info(f"load model to {device} done!")

min_loss = 0
idx = 0
for round in range(config.epoch):
    for train_data_batch in train_dataloader:
        model.train()
        # optimizer.clear_grad()
        optimizer.zero_grad()
        id_list, padding_mask, item_list, country_list = train_data_batch
        id_list, padding_mask, item_list, country_list = \
            id_list.to(device), padding_mask.to(device), item_list.to(device), country_list.to(device)
        if config.log_level >= 1:
            logging.info(f"id_list.shape:{id_list.shape}, padding_mask:{padding_mask.shape}, item_list.shape:{item_list.shape}, country_list.shape:{country_list.shape}")
        user_vec, item_vec, loss = model(id_list, padding_mask, country_list=country_list, item_list=item_list)

        loss.backward()
        optimizer.step()
        if idx % config.log_interval == 0:
            logging.info(f'{idx:5d}th step | loss {loss.cpu().item():5.6f}')
        if idx % config.eval_step == 0:
            model.eval()
            test_loss_sum = 0
            test_cnt = 0
            for test_data_batch in test_dataloader:
                test_id_list, test_padding_mask, test_item_list, test_country_list = test_data_batch
                test_id_list, test_padding_mask, test_item_list, test_country_list = \
                    test_id_list.to(device), test_padding_mask.to(device), test_item_list.to(device), test_country_list.to(device)
                _, _, test_loss = model(test_id_list, test_padding_mask, country_list=test_country_list, item_list=test_item_list)
                test_loss_sum += test_loss.cpu().item()
                test_cnt += 1
            test_loss_avg = test_loss_sum/test_cnt

            logging.info(
                    f'{idx:5d}th step | test_loss_avg {test_loss_avg:5.6f}')

        if idx >= config.eval_step and (min_loss > test_loss_avg or idx % config.save_step == 0):
            if min_loss > test_loss_avg:
                min_loss = test_loss_avg
            model_name = f"{config.save_path}/u2i_v{version}_steps_{idx}_{test_loss_avg:.4f}.model"
            torch.save(model.state_dict(), model_name)
            logging.info(f"model {model_name} is saved!")
        idx += 1
    scheduler.step()

model_name = f"{config.save_path}/u2i_v{version}_steps_{idx}_{test_loss_avg:.4f}.model"
torch.save(model.state_dict(), model_name)
logging.info(f"model {model_name} is saved!")
