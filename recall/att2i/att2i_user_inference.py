import sys
sys.path.append("../..")
import torch
from torch.optim.lr_scheduler import StepLR
import numpy as np
import random
from att2i_model import AttentionU2I
from utils.args import config
from u2i_dataloader.test_dataloader import UserDataset
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"device:{device}")

user_dataset = UserDataset(file=config.test_file, max_seq_len=config.max_seq_len)
user_dataloader = DataLoader(user_dataset, config.test_batch_size)

model.to(device)
logging.info(f"load model to {device} done!")
model.eval()

fd = open(config.output_file, "w")
idx = 0
for user_data_batch in user_dataloader:
    if idx % 10000 == 0:
        logging.info(f"{idx} batches inference done!")
    id_list, padding_mask, item_list, country_list = user_data_batch
    id_list, padding_mask, country_list = \
        id_list.to(device), padding_mask.to(device), country_list.to(device)
    user_vec = model(id_list, padding_mask, country_list=country_list, item_list=None, type=1)
    user_vec_arr = list(user_vec.detach().squeeze().cpu().numpy())
    item_list_arr = list(item_list.detach().squeeze().cpu().numpy())
    country_list_arr = list(country_list.detach().squeeze().cpu().numpy())
    for i in range(len(user_vec_arr)):
        item_id = item_list_arr[i]
        user_vec_str = ",".join([str(ele) for ele in user_vec_arr[i]])
        fd.write(f"{user_vec_str}\t{country_list_arr[i]}\t{item_id}\n")
    idx += 1
fd.close()