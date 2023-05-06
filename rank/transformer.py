import torch.nn as nn
import torch
import sys
sys.path.append("..")
from utils.feed_forward import PositionwiseFeedForward
from utils.multi_head_attention import MultiHeadedAttention
from utils.multi_head_attention import clones
from utils.position import LearnedPositionalEncoding
import copy


class LayerNorm(nn.Module):
    """
    Layer normalization module.
    """
    def __init__(self, features, eps=1e-6):
        """
        :param features: shape of normalized features
        :param eps: epsilon used for standard deviation
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))  # type: ignore
        self.b_2 = nn.Parameter(torch.zeros(features))  # type: ignore
        self.eps = eps

    def forward(self, x):
        """
        Forward pass through the layer normalization.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :return: normalized input of shape [batch_size, slate_length, output_dim]
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    Residual connection followed by layer normalization.
    Please not that for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        """
        :param size: number of input/output features
        :param dropout: dropout probability
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Forward pass through the sublayer connection module, applying the residual connection to any sublayer with the same size.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :param sublayer: layer through which to pass the input prior to applying the sum
        :return: output of shape [batch_size, slate_length, output_dim]
        """
        return x + self.dropout(
            sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """
    Single Transformer encoder block made of self-attention and feed-forward layers with residual connections.
    """
    def __init__(self, emb_size, self_attn, feed_forward, dropout):
        """
        :param size: input/output size of the encoder block
        :param self_attn: self-attention layer
        :param feed_forward: feed-forward layer
        :param dropout: dropout probability
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = dropout
        self.emb_size = emb_size
        self.sublayer = clones(SublayerConnection(self.emb_size, self.dropout), 2)

    def forward(self, x, mask):
        """
        Forward pass through the encoder block.
        :param x: input of shape [batch_size, slate_length, self.size]
        :param mask: padding mask of shape [batch_size, slate_length]
        :return: output of shape [batch_size, slate_length, self.size]
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Transformer(nn.Module):
    def __init__(self, num_layers, num_head, d_model, d_ff, max_len, dropout=0.1):
        super(Transformer, self).__init__()
        self.num_layers = num_layers
        self.num_head = num_head
        self.d_model = d_model
        self.d_ff = d_ff  # dimension of hidden layer of feed_forward
        self.dropout = dropout
        multihead_attention = MultiHeadedAttention(self.num_head, self.d_model, self.dropout)
        feed_forward = PositionwiseFeedForward(self.d_model, self.d_ff)
        layer = EncoderLayer(self.d_model, copy.deepcopy(multihead_attention), copy.deepcopy(feed_forward), dropout)
        self.layers = clones(layer, self.num_layers)
        self.norm = LayerNorm(layer.emb_size)
        self.max_len = max_len
        self.positional_encoding = LearnedPositionalEncoding(self.d_model, self.max_len)

    def forward(self, x, mask):
        input = self.positional_encoding(x, mask)

        for layer in self.layers:
            input = layer(input, mask)

        return self.norm(input)


if __name__ == "__main__":
    tfm = Transformer(3, 4, 32, 40, 128)
    input = torch.ones((10, 20, 32))
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
    output = tfm(input, mask)
    print(output.shape)
