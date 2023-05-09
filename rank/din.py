import torch.nn as nn
import torch
import sys
sys.path.append("..")
from model_utils.fully_connected_layer import FullyConnectedLayer


class DeepInterestNetwork(nn.Module):
    def __init__(self, emb_dim, att_hidden_size=[80, 40],
                 attn_bias=[True, True], att_batch_norm=False):
        super(DeepInterestNetwork, self).__init__()
        # self.max_len = max_len
        self.emb_dim = emb_dim
        # self.item_emb = nn.Embedding(vocab_size, emb_dim)

        self.att_fc1 = FullyConnectedLayer(input_size=4 * self.emb_dim,
                                           hidden_size=att_hidden_size,
                                           bias=attn_bias,
                                           batch_norm=att_batch_norm,
                                           activation='dice',
                                           dice_dim=3)

        self.att_fc2 = FullyConnectedLayer(input_size=att_hidden_size[-1],
                                           hidden_size=[1],
                                           bias=[True],
                                           batch_norm=att_batch_norm,
                                           activation='dice',
                                           dice_dim=3)


    def forward(self, keys, queries, mask):
        """
        keys: [batch_size, slate_len, emb_size]
        queries: [batch_size, 1, emb_size]
        mask: [batch_size, slate_len] 需要的位置为0，不需要的位置为0，与transformer的mask格式对齐
        """

        user_behavior_len = keys.size(1)
        queries = torch.cat([queries for _ in range(user_behavior_len)], dim=1)
        attention_input = torch.cat([queries, keys, queries - keys, queries * keys], dim=-1)

        attention_output = self.att_fc1(attention_input)
        attention_output = self.att_fc2(attention_output)  # [batch_size, slate_len, 1]
        attention_score = torch.transpose(attention_output, 1, 2)  # [batch_size, 1, slate_len]
        # print("attention_score.shape:", attention_score.shape)

        # user_behavior_length = user_behavior_length.long()
        # mask = torch.arange(keys.size(1))[None, :] < user_behavior_length[:, None]
        mask = 1 - mask.unsqueeze(1)
        # print(mask)

        # mask
        output = torch.mul(attention_score, mask.float())  # batch_size *
        # print(mask.shape, output.shape)

        # multiply weight
        # print(output.shape, keys.shape)
        output = torch.matmul(output, keys)

        return output


if __name__ == "__main__":
    din = DeepInterestNetwork(4)
    query = torch.ones((32, 1, 4))
    keys = torch.ones((32, 20, 4))
    # lens = torch.ones((32, 1))
    mask = torch.tensor([
        [0] * 10 + [1] * 10,
        [0] * 20 + [1] * 0,
        [0] * 15 + [1] * 5,
        [0] * 12 + [1] * 8,
        [0] * 5 + [1] * 15,
        [0] * 4 + [1] * 16,
        [0] * 7 + [1] * 13,
        [0] * 12 + [1] * 8,
        [0] * 3 + [1] * 17,
        [0] * 6 + [1] * 14,
        [0] * 10 + [1] * 10,
        [0] * 20 + [1] * 0,
        [0] * 15 + [1] * 5,
        [0] * 12 + [1] * 8,
        [0] * 5 + [1] * 15,
        [0] * 4 + [1] * 16,
        [0] * 7 + [1] * 13,
        [0] * 12 + [1] * 8,
        [0] * 3 + [1] * 17,
        [0] * 6 + [1] * 14,
        [0] * 10 + [1] * 10,
        [0] * 20 + [1] * 0,
        [0] * 15 + [1] * 5,
        [0] * 12 + [1] * 8,
        [0] * 5 + [1] * 15,
        [0] * 4 + [1] * 16,
        [0] * 7 + [1] * 13,
        [0] * 12 + [1] * 8,
        [0] * 3 + [1] * 17,
        [0] * 6 + [1] * 14,
        [0] * 7 + [1] * 13,
        [0] * 12 + [1] * 8
    ])

    att = din(keys, query, mask)
    print(att.shape)





