import math

import torch
from torch import nn

class LearnablePositionEncoder(nn.Module):
    def __init__(self, d_model, dropout=0, max_len=5):
        super(LearnablePositionEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dim = d_model

    def forward(self, x):
        # x [bs, max_len, d_model]
        x = x * math.sqrt(self.dim)
        x = x + self.position_embedding.weight
        return self.dropout(x)


if __name__ == '__main__':
    d_model = 10
    encoder = LearnablePositionEncoder(d_model)
    input = torch.rand(2, 5, d_model)
    encoder(input)