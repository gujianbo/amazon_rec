import sys
sys.path.append("../..")
import torch
from torch.optim.lr_scheduler import StepLR
import numpy as np
import random
from att2i_model import AttentionU2I
from utils.args import config
from u2i_dataloader.test_dataloader import ItemDataset
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

item_dataset = ItemDataset(file=config.test_file, max_seq_len=config.max_seq_len)
item_dataloader = DataLoader(item_dataset, config.test_batch_size)

model.to(device)
logging.info(f"load model to {device} done!")
model.eval()

fd = open(config.output_file, "w")
idx = 0
for item_data_batch in item_dataloader:
    if idx % 1000 == 0:
        logging.info(f"{idx} batches inference done!")
    item_list = item_data_batch
    item_list, country_list = item_list.to(device)
    item_vec = model(None, None, country_list=None, item_list=item_list, type=2)
    item_vec_arr = list(item_vec.detach().squeeze().cpu().numpy())
    item_list_arr = list(item_list.detach().squeeze().cpu().numpy())
    for i in range(len(item_vec_arr)):
        item_id = item_list_arr[i]
        item_vec_str = ",".join([str(ele) for ele in item_vec_arr[i]])
        fd.write(f"{item_id}\t{item_vec_str}\n")
    idx += 1
fd.close()