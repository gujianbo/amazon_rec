import math
import torch
import torch.nn as nn


class FixedPositionalEncoding(nn.Module):
    """
    Class implementing fixed positional encodings.

    Fixed positional encodings up to max_len position are computed once during object construction.
    """
    def __init__(self, d_model: int, max_len=5000):
        """
        :param d_model: dimensionality of the embeddings
        :param max_len: maximum length of the sequence
        """
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = torch.cat((pe, torch.zeros([1, d_model])))
        self.padding_idx = pe.size()[0] - 1
        self.register_buffer('pe', pe)

    def forward(self, x, mask):
        """
        Forward pass through the FixedPositionalEncoding.
        :param x: input of shape [batch_size, slate_length, d_model]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: output of shape [batch_size, slate_length, d_model]
        """
        max_len = x.shape[1]
        batch_size = x.shape[1]
        indices = torch.arange(max_len).unsqueeze(0).repeat(batch_size, 1)

        padded_indices = indices.masked_fill(mask == 1, self.padding_idx)
        padded_indices[padded_indices > self.padding_idx] = self.padding_idx
        x = math.sqrt(self.pe.shape[1]) * x + self.pe[padded_indices, :]  # type: ignore
        return x


class LearnedPositionalEncoding(nn.Module):
    """
    Class implementing learnable positional encodings.
    """
    def __init__(self, d_model, max_len=5000):
        """
        :param d_model: dimensionality of the embeddings
        :param max_len: maximum length of the sequence
        """
        super().__init__()

        self.pe = nn.Embedding(max_len + 1, d_model, padding_idx=-1)

    def forward(self, x, mask):
        """
        Forward pass through the LearnedPositionalEncoding.
        :param x: input of shape [batch_size, slate_length, d_model]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: output of shape [batch_size, slate_length, d_model]
        """
        max_len = x.shape[1]
        batch_size = x.shape[0]
        indices = torch.arange(max_len).unsqueeze(0).repeat(batch_size, 1).to(x.device)
        # print(indices.shape, indices)
        # print(mask.shape, mask)

        padded_indices = indices.masked_fill(mask == 1, self.pe.padding_idx)
        padded_indices[padded_indices > self.pe.padding_idx] = self.pe.padding_idx
        # print(padded_indices.shape, padded_indices)
        x = math.sqrt(self.pe.embedding_dim) * x + self.pe(padded_indices)
        return x

if __name__ == "__main__":
    input = torch.ones((10, 20, 32))
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
        [0] * 6 + [1] * 14
    ])

    pe = LearnedPositionalEncoding(32, 128)
    ec = pe(input, mask)
    print(ec)