import torch.nn as nn
import torch

from rank.transformer import Transformer
import torch.nn.functional as F


class AttentionU2I(nn.Module):
    def __init__(self, num_items, num_country, num_layers, num_head, d_model, d_ff, max_len, dropout,
                 input_size, hidden_size, emb_size, temperature=1.0):
        super(AttentionU2I, self).__init__()
        self.num_items = num_items
        # self.num_types = num_types
        self.num_layers = num_layers
        self.num_head = num_head
        self.d_model = d_model
        self.d_ff = d_ff
        self.max_len = max_len
        self.dropout = dropout
        self.transformer = Transformer(num_layers, num_head, d_model, d_ff, max_len, dropout)
        self.item_embedding = nn.Embedding(num_items, d_model)
        self.country_embedding = nn.Embedding(num_country, d_model)
        # self.type_embedding = nn.Embedding(num_types, d_model)

        self.user_tower_fc1 = nn.Linear(input_size, hidden_size)
        self.user_tower_bn1 = nn.BatchNorm1d(hidden_size)
        self.user_tower_fc2 = nn.Linear(hidden_size, emb_size)
        self.user_tower_bn2 = nn.BatchNorm1d(emb_size)
        self.relu = nn.ReLU()

        self.item_tower = nn.Embedding(num_items, d_model)
        self.item_tower_fc1 = nn.Linear(d_model, hidden_size)
        self.item_tower_bn1 = nn.BatchNorm1d(hidden_size)
        self.item_tower_fc2 = nn.Linear(hidden_size, emb_size)
        self.item_tower_bn2 = nn.BatchNorm1d(emb_size)

        self.dropout_net = nn.Dropout(p=self.dropout)
        # if prerank_logits == 1:
        #     self.prerank_logits = True
        #     self.loss = nn.BCELoss(reduction='none')
        # else:
        #     self.prerank_logits = False
        # if top_cross == 1:
        #     self.top_tower_fc1 = nn.Linear(2*emb_size, emb_size)
        #     self.top_tower_bn1 = nn.BatchNorm1d(emb_size)
        #     if self.prerank_logits:
        #         self.top_tower_fc2 = nn.Linear(emb_size, 3)
        #         self.sig = nn.Sigmoid()
        #     else:
        #         self.top_tower_fc2 = nn.Linear(emb_size, 1)
        self.ce = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, id_list, mask, other_feat=None, country_list=None, item_list=None, type=0):
        if type == 2:  # item
            i_vec = self.item_vec(item_list, country_list)
            return i_vec
        id_list_feat = self.item_embedding(id_list)
        seq_len = id_list.shape[1]
        country_feat = self.country_embedding(country_list)
        # country_repeat_feat = country_feat.repeat([1, seq_len, 1])
        # id_list_feat += country_repeat_feat
        # type_list_feat = self.type_embedding(type_list)
        # id_list_feat += type_list_feat

        # tfm_feat = self.transformer(id_list_feat, mask)

        new_mask = mask.unsqueeze(2).repeat(1, 1, self.d_model)
        # tfm_feat = tfm_feat.masked_fill(new_mask == 1, 0.0)
        # print(f"tfm_feat:{tfm_feat}, tfm_feat.shape:{tfm_feat.shape}")

        # tfm_sum = torch.sum(tfm_feat, dim=1)
        # mask_sum = torch.sum(1-new_mask, dim=1) + 0.00000001
        # tfm_mean = tfm_sum / mask_sum

        id_list_feat = id_list_feat.masked_fill(new_mask == 1, 0.0)
        seq_feat = torch.sum(id_list_feat, dim=1)
        mask_sum = torch.sum(1 - new_mask, dim=1) + 0.00000001
        seq_mean = seq_feat / mask_sum

        # batch_size = id_list.shape[0]
        # transf_feat = torch.reshape(tfm_feat, (batch_size, -1))
        # tfm_cls = tfm_feat[:, 0, :]

        batch_size = country_feat.shape[0]
        country_feat_fla = torch.reshape(country_feat, (batch_size, -1))

        if other_feat is None:
            all_feat = torch.concat([seq_mean, country_feat_fla], dim=1)
        else:
            all_feat = torch.concat([seq_mean, country_feat_fla, other_feat], dim=1)

        user_vec = self.user_tower_fc1(all_feat)
        user_vec = self.user_tower_bn1(user_vec)
        user_vec = self.relu(user_vec)
        user_vec = self.dropout_net(user_vec)
        user_vec = self.user_tower_fc2(user_vec)
        user_vec = self.user_tower_bn2(user_vec)
        user_vec = F.normalize(user_vec, dim=-1)

        if type == 1:  # user
            return user_vec
        elif item_list is not None:
            i_vec = self.item_vec(item_list, country_list)
            # print(f"i_vec.shape:{i_vec.shape}")
            # i_vec = i_vec.squeeze()
            batch_size = user_vec.shape[0]
            label = torch.arange(batch_size).to(id_list.device)
            sim = user_vec.mm(i_vec.transpose(0, 1)) / self.temperature
            # print("sim.shape:", sim.shape)
            # print("label.shape:", label.shape)
            loss = self.ce(sim, label)
            # print("loss:", loss)

            # item_len = item_list.shape[1]
            # user_vec = user_vec.unsqueeze(1).repeat(1, item_len, 1)
            #
            # feat = torch.concat([user_vec, i_vec], dim=2)
            # feat = torch.reshape(feat, [batch_size * item_len, -1])
            # feat = self.top_tower_fc1(feat)
            # feat = self.top_tower_bn1(feat)
            # feat = self.dropout_net(feat)
            # feat = self.relu(feat)
            # score = self.top_tower_fc2(feat)
            # score = self.sig(score)
            # if label is None:
            #     out = torch.reshape(score, [batch_size, item_len, -1])
            #     return out
            # label_list = torch.reshape(label, [batch_size * item_len, -1])
            # candi_mask_list = torch.reshape(candi_mask, [batch_size * item_len, -1]).repeat((1, 3))
            # bce_mat = self.loss(score, label_list)
            # loss = torch.sum(bce_mat * candi_mask_list)/torch.sum(candi_mask_list)
            # out = torch.reshape(score, [batch_size, item_len, -1])
            return user_vec, i_vec, loss
        else:
            return None


    def item_vec(self, item_list, country_list):
        item_vec = self.item_tower(item_list.squeeze())
        # print(f"item_vec.shape:{item_vec.shape}")
        # country_vec = self.country_embedding(country_list.squeeze())
        # item_vec = torch.concat([item_vec, country_vec], dim=1)

        # batch_size = item_vec.shape[0]
        # seq_len = item_vec.shape[1]
        # item_vec = torch.reshape(item_vec, [batch_size * seq_len, -1])
        item_vec = self.item_tower_fc1(item_vec)
        item_vec = self.item_tower_bn1(item_vec)
        item_vec = self.relu(item_vec)
        item_vec = self.dropout_net(item_vec)
        item_vec = self.item_tower_fc2(item_vec)
        item_vec = self.item_tower_bn2(item_vec)
        item_vec = F.normalize(item_vec, dim=-1)
        # print(f"item_vec.shape:{item_vec.shape}")

        # item_vec = torch.reshape(item_vec, [batch_size, seq_len, -1])
        return item_vec


if __name__ == "__main__":
    att = AttentionU2I(num_items=100, num_country=7, num_layers=2, num_head=4,
                       d_model=256, d_ff=256, max_len=20, dropout=0.1,
                       input_size=5376+256, hidden_size=128, emb_size=64)
    id_list = torch.tensor([
        [4, 3, 67, 34, 93, 23, 12, 3, 7, 2, 9, 10, 25, 37, 47, 0, 0, 0, 0, 0],
        [25, 3, 67, 37, 93, 24, 12, 3, 7, 2, 84, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [65, 45, 22, 63, 26, 45, 25, 95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [76, 87, 34, 26, 77, 34, 64, 39, 50, 78, 89, 28, 39, 66, 0, 0, 0, 0, 0, 0],
        [35, 99, 38, 26, 4, 25, 37, 46, 26, 35, 31, 80, 37, 62, 16, 28, 36, 76, 25, 55],
        [35, 36, 15, 35, 26, 3, 4, 26, 48, 94, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 25, 15, 25, 44, 35, 42, 27, 58, 27, 90, 89, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    country_list = torch.tensor([
        [1], [2], [3], [1], [0], [1], [5]
    ])

    # type_list = torch.tensor([
    #     [1, 3, 2, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 1, 1, 0, 0, 0, 0, 0],
    #     [1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [1, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    #     [1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2],
    #     [1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    # ])

    mask = torch.tensor([
        [0] * 15 + [1] * 5,
        [0] * 11 + [1] * 9,
        [0] * 8 + [1] * 12,
        [0] * 14 + [1] * 6,
        [0] * 20,
        [0] * 10 + [1] * 10,
        [0] * 12 + [1] * 8
    ])

    # item = torch.tensor([
    #     [1, 3, 2, 1],
    #     [1, 1, 1, 1],
    #     [1, 1, 1, 1],
    #     [1, 1, 2, 1],
    #     [1, 1, 2, 3],
    #     [1, 1, 1, 2],
    #     [1, 1, 1, 1]
    # ])
    item = torch.tensor([
        [2], [3], [1], [4], [5], [6], [7]
    ])

    user_vec, i_vec, loss = att(id_list, mask, None, country_list, item_list=item)
    print("user_vec.shape:", user_vec.shape, user_vec)
    print("i_vec.shape:", i_vec.shape, i_vec)
    print("loss:", loss)
    # item_vec = att.item_vec(item)
    # print(f"user_vec:{user_vec}, user_vec.shape:{user_vec.shape}")
    # print(f"item_vec:{item_vec}, item_vec.shape:{item_vec.shape}")
    #
    # print(f"user_vec.unsqueeze(1).shape:{user_vec.unsqueeze(1).shape}")
    # print(f"item_vec.transpose(-2, -1).shape:{item_vec.transpose(-2, -1).shape}")
    # product = torch.matmul(user_vec.unsqueeze(1), item_vec.transpose(-2, -1))
    # print(f"product:{product}, product.shape:{product.shape}")
