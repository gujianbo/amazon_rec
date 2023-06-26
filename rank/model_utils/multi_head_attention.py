import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import copy


def clones(module, N):
    """
    Creation of N identical layers.
    :param module: module to clone
    :param N: number of copies
    :return: nn.ModuleList of module copies
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    """
    Multi-headed attention block.
    """
    def __init__(self, num_head, d_model, dropout=0.1):
        """
        :param h: number of attention heads
        :param d_model: input/output dimensionality
        :param dropout: dropout probability
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_head == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // num_head
        self.num_head = num_head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass through the multi-head attention block.
        :param query: query set of shape [batch_size, slate_size, self.d_model]
        :param key: key set of shape [batch_size, slate_size, self.d_model]
        :param value: value set of shape [batch_size, slate_size, self.d_model]
        :param mask: padding mask of shape [batch_size, slate_length]
        :return: output of shape [batch_size, slate_size, self.d_model]
        """
        if mask is not None:
            # same mask applied to all h heads
            max_len = query.size(1)
            # print(max_len)
            mask = mask.unsqueeze(1).unsqueeze(1).repeat(1, self.num_head, max_len, 1)
            # print(mask)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [linear(x).view(nbatches, -1, self.num_head, self.d_k).transpose(1, 2)
             for linear, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.num_head * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None, dropout=None):
        """
        Basic function for "Scaled Dot Product Attention" computation.
        :param query: query set of shape [batch_size, slate_size, n_attention_heads, attention_dim]
        :param key: key set of shape [batch_size, slate_size, n_attention_heads, attention_dim]
        :param value: value set of shape [batch_size, slate_size, n_attention_heads, attention_dim]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param dropout: dropout probability
        :return: attention scores of shape [batch_size, slate_size, n_attention_heads, attention_dim]
        """
        d_k = query.size(-1)
        # print(f"query.shape:{query.shape}")
        # print(f"key.transpose(-2, -1).shape:{key.transpose(-2, -1).shape}")
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # print("scores.shape", scores.shape)
        # print("mask.shape:", mask.shape)

        if mask is not None:
            scores = scores.masked_fill(mask == 1, float("-inf"))

        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


if __name__ == "__main__":
    mha = MultiHeadedAttention(4, 32)
    input = torch.rand((10, 20, 32))
    mask = torch.tensor([
        [0]*10+[1]*10,
        [0]*20+[1]*0,
        [0]*15+[1]*5,
        [0]*12+[1]*8,
        [0]*5+[1]*15,
        [0]*4+[1]*16,
        [0]*7+[1]*13,
        [0]*12+[1]*8,
        [0]*3+[1]*17,
        [0]*6+[1]*14
    ])
    output = mha(input, input, input, mask)
    print(output.shape)
    # mask = torch.tensor([
    #     [0]*10+[1]*20,
    #     [0]*20+[1]*10,
    #     [0]*15+[1]*15,
    #     [0]*28+[1]*2
    # ])
    # print(mask.dim())
    # mask = mask.unsqueeze(1).repeat(1, 4, 1)
    # print(mask)
    import d2l
