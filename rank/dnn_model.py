import torch.nn as nn
import torch
from transformer import Transformer
from din import DeepInterestNetwork
from utils.mlp import Tower


class DNNModel(nn.Module):
    def __init__(self, num_items, d_model, num_layers, num_head, d_ff, max_len, dropout):
        super(DNNModel, self).__init__()
        self.d_model = d_model
        self.item_embedding = nn.Embedding(num_items, d_model)
        self.transformer = Transformer(num_layers, num_head, d_model, d_ff, max_len, dropout)
        self.din = DeepInterestNetwork(d_model)
        self.ts_embedding = nn.Embedding(1, d_model)
        self.mlp = Tower(512, 128, 1)

    def forward(self, id_list, ts_list, candi, mask, other_feat):
        id_list_feat = self.item_embedding(id_list)
        # 时间delta特征
        ts_zeros = torch.zeros_like(ts_list, dtype=torch.int32).to(ts_list.device)
        ts_emb = self.ts_embedding(ts_zeros)
        ts_weight = ts_list.unsqueeze(2).repeat([1, 1, self.d_model])
        ts_emb = ts_emb * ts_weight
        id_list_feat += ts_emb

        tfm_feat = self.transformer(id_list_feat, mask)

        new_mask = mask.unsqueeze(2).repeat(1, 1, self.d_model)
        tfm_feat = tfm_feat.masked_fill(new_mask == 1, 0.0)
        id_feat = self.item_embedding(candi)
        din_feat = self.din(tfm_feat, id_feat, mask)
        batch_size = id_list.shape[0]
        # print(batch_size)
        # print("tfm_feat.shape", tfm_feat.shape)
        # print("din_feat.shape", din_feat.shape)
        # print("id_feat.shape", id_feat.shape)

        transf_feat = torch.reshape(tfm_feat, (batch_size, -1))
        deep_interest_net_feat = torch.reshape(din_feat, (batch_size, -1))
        id_emb_feat = torch.reshape(id_feat, (batch_size, -1))
        all_feat = torch.concat([transf_feat, deep_interest_net_feat, id_emb_feat, other_feat], dim=1)
        logits = self.mlp(all_feat)
        return logits

