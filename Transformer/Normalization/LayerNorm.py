import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-9):
        super().__init__()
        self.size = d_model
        self.eps = eps
        # layernorm中每个特征都有独立的缩放因子和偏移量！
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

    def forward(self, x):
        # 在最后一个维度(特征维度)上计算均值和方差，keepdim=True保持原来的维度
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


if __name__ == '__main__':
    x = LayerNorm(10)
    input = torch.rand(2, 5, 10)
    print(x(input))