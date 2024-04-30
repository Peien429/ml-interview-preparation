import torch

from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-9):
        super().__init__()
        self.size = d_model
        self.alpha = torch.ones(d_model)
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * x / (torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True)) + self.eps)
        return norm