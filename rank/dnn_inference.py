
import torch
import torch.nn as nn
import numpy as np
import random
import sys
sys.path.append("..")
from dnn_model import DNNModel
from utils.args import config
from dataloader.test_dataloader import SubmissionDatasetList
from torch.utils.data import DataLoader

import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)

random.seed(config.seed+1)
random.seed(config.seed)
np.random.seed(config.seed)
print(config)

input_size = (config.max_seq_len + 3) * config.d_model + config.dense_size
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
    model_dict = model.state_dict()
    ptm = torch.load(config.init_parameters)
    state_dict = {k: v for k, v in ptm.items() if k in model_dict.keys()}

    for k, v in state_dict.items():
        logging.debug(f"loading {k}, shape:{v.shape}")
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
else:
    logging.error('no init_parameters!')
    exit(1)

test_dataset = SubmissionDatasetList(config.test_file, need_label=True, max_seq_len=config.max_seq_len, local_code=[1, 2, 3])
test_dataloader = DataLoader(test_dataset, config.test_batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"device:{device}")
model.to(device)
logging.info(f"load model to {device} done!")


model.eval()
fdout = open(config.output_file, "w")
idx = 0
for test_data_batch in test_dataloader:
    if idx % 10000 == 0:
        logging.info(f"{idx} batches inference done!")
    test_prev_ids, test_padding_mask, test_locale_code, test_dense_feat, test_candi_id, test_label = test_data_batch
    test_prev_ids, test_padding_mask, test_locale_code, test_dense_feat, test_candi_id = \
        test_prev_ids.to(device), test_padding_mask.to(device), test_locale_code.to(device), test_dense_feat.to(
            device), test_candi_id.to(device)

    logits_test = model(test_prev_ids, test_padding_mask, test_locale_code, test_candi_id, test_dense_feat)

    test_prev_ids_arr = list(test_prev_ids.detach().squeeze().cpu().numpy())
    test_locale_code_arr = list(test_locale_code.detach().squeeze().cpu().numpy())
    test_candi_id_arr = list(test_candi_id.detach().squeeze().cpu().numpy())
    test_logits = list(logits_test.detach().squeeze().cpu().numpy())
    test_label_t = torch.tensor(test_label, dtype=torch.float32).to(device)
    test_labels = list(test_label_t.detach().cpu().numpy())

    for i in range(len(test_prev_ids_arr)):
        prev_ids = test_prev_ids_arr[i]
        prev_ids_str = ",".join([str(item) for item in prev_ids if item != 0])
        candi_id = test_candi_id_arr[i]
        locale_code = test_locale_code_arr[i]
        logit = test_logits[i]
        label = test_labels[i]

        fdout.write(f"{prev_ids_str}\t{candi_id}\t{locale_code}\t{label}\t{logitßß}\n")
    idx+=1
fdout.close()
logging.info("inference done!")


