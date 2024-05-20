import math

import torch
import torch.nn.functional as F
from torch import nn


class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.3):
        super().__init__()
        assert d_model % n_head == 0
        # 基本数据：模型维度，注意力头数，每个头的维度，缩放因子
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.scale = math.sqrt(self.d_k)
        # 注意力分数随即丢弃
        self.dropout = nn.Dropout(p=dropout)
        # 4个线性层
        # 所有注意力头共用Key/Value，每个注意力头独享Query，维度均为d_k
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, self.d_k)
        self.v_linear = nn.Linear(d_model, self.d_k)
        # 最后的线性层的维度和多头注意力的实现相比没有变化
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # key和value的长度相同，但是query的长度不一定和key/value相同

        bs = q.shape[0]
        # Query的线性变换，然后把隐藏层拆成多个头，然后把头放到第二维
        query = self.q_linear(q).view(bs, -1, self.n_head, self.d_k).transpose(1, 2)
        # Key/Value的线性变换，维度变成[bs, seq_len, d_k]；增加注意力头对应的维度，后面利用广播实现多个注意力头共享Key/Value
        key = self.k_linear(k).unsqueeze(1)
        value = self.v_linear(v).unsqueeze(1)
        # 对key求转置，然后计算乘积，然后除以缩放因子
        attention_score = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        # mask之后在最后一个维度做softmax，随即丢弃注意力分数
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)
        attention_score = self.dropout(F.softmax(attention_score, dim=-1))
        # 注意力分数乘以value，然后把seq放到第二维，然后合并多个头
        # 必须使用contiguous()，否则view会报错
        output = torch.matmul(attention_score, value).transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(output)
        return output


if __name__ == '__main__':
    mha = MultiQueryAttention(d_model=10, n_head=2)
    q = torch.rand(1, 7, 10)
    k = torch.rand(1, 3, 10)
    v = torch.rand(1, 3, 10)
    print(mha(q, k, v))


