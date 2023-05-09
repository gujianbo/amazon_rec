import torch.nn as nn
import torch
from transformer import Transformer
from din import DeepInterestNetwork
from model_utils.mlp import Tower
import sys
sys.path.append("..")
from utils.args import config

import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)


class DNNModel(nn.Module):
    def __init__(self, input_size, num_items, d_model, num_layers, num_head, d_ff, max_len, dropout):
        super(DNNModel, self).__init__()
        self.d_model = d_model
        self.item_embedding = nn.Embedding(num_items, d_model)
        self.transformer = Transformer(num_layers, num_head, d_model, d_ff, max_len, dropout)
        self.din = DeepInterestNetwork(d_model)
        self.ts_embedding = nn.Embedding(1, d_model)
        self.input_size = input_size
        self.mlp = Tower(input_size, 128, 1)

    def forward(self, id_list, mask, locale_code, candi, other_feat):
        id_list_feat = self.item_embedding(id_list)

        tfm_feat = self.transformer(id_list_feat, mask)

        new_mask = mask.unsqueeze(2).repeat(1, 1, self.d_model)
        tfm_feat = tfm_feat.masked_fill(new_mask == 1, 0.0)
        id_feat = self.item_embedding(candi)
        # logging.info(f"tfm_feat.shape:{tfm_feat.shape} | id_feat.shape:{id_feat.shape} | mask:{mask.shape}")
        din_feat = self.din(tfm_feat, id_feat, mask)
        batch_size = id_list.shape[0]
        # print(batch_size)
        # print("tfm_feat.shape", tfm_feat.shape)
        # print("din_feat.shape", din_feat.shape)
        # print("id_feat.shape", id_feat.shape)

        transf_feat = torch.reshape(tfm_feat, (batch_size, -1))
        deep_interest_net_feat = torch.reshape(din_feat, (batch_size, -1))
        id_emb_feat = torch.reshape(id_feat, (batch_size, -1))
        all_feat = torch.concat([transf_feat, deep_interest_net_feat, id_emb_feat, locale_code, other_feat], dim=1)
        logits = self.mlp(all_feat)
        return logits

