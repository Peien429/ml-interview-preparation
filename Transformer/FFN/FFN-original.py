import torch
from torch import nn

class FFN(nn.Module):
    def __init__(self, d_model, d_hidden, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
