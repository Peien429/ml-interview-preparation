import math
import torch
from torch import nn


class SinPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0, max_len=5):
        super(SinPositionalEncoding, self).__init__()
        # dropout用于对位置编码和词嵌入的和做丢弃
        self.dropout = torch.nn.Dropout(dropout)
        # sin位置编码是固定的，可以直接初始化，不用每次forward都计算
        pe = torch.zeros(max_len, d_model)
        # 第一步，生成pos，并且在第一维上增加维度，后续广播到d_model维度
        position = torch.arange(0, max_len).unsqueeze(1)
        # 第二步，生成pos以外的部分，并且这里pos与其做乘法
        # 这里是某个官方代码实现，它相当于先取了ln并化简，然后再exp
        # exp(ln(10000) * (-2i/d_model)) = 10000^(-2i/d_model) = (1/10000)^(2i/d_model)
        # exp和log都默认以e为底
        div_term = torch.exp(
            -(torch.arange(0, d_model, 2) / d_model) * (math.log(10000))
        )
        # 第三步，生成sin和cos，这里是乘法
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 第四步，在第零维上增加维度，因为输入可能是成Batch的，增加维度利用广播机制为成Batch的输入增加位置编码
        pe = pe.unsqueeze(0)
        # register_buffer的作用是将pe注册为模型的参数；如果没有这行代码，那么pe将被视为对象属性，而不是模型的参数
        self.register_buffer("pe", pe)
        self.dim = d_model

    def forward(self, x):
        # x [bs, max_len, d_model]
        # 这一步是为了放大词嵌入向量的影响
        x = x * math.sqrt(self.dim)
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


class SinPositionalEncoding2(nn.Module):
    def __init__(self, d_model, dropout=0, max_len=5):
        super(SinPositionalEncoding2, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        # 个人感觉这种实现方式更加直观，但可能因为除法导致可能存在问题，所以官方实现才使用的乘法
        # 首先是将pos视为分子，其他部分视为分母，分母相当于以10000为底的指数
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.pow(10000, torch.arange(0, d_model, 2) / d_model)
        # 这里是除法
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        self.dim = d_model

    def forward(self, x):
        x = x * math.sqrt(self.dim)
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


if __name__ == '__main__':
    d_model = 10
    encoder_1 = SinPositionalEncoding(d_model)
    encoder_2 = SinPositionalEncoding2(d_model)
    bs = 2
    max_len = 5
    input = torch.rand(bs, max_len, d_model)
    print(encoder_1(input))
    print(encoder_2(input))
    # allclose是判断两个tensor是否相等的函数，但是允许有一定的误差，这个永远是True
    print(torch.allclose(encoder_1(input), encoder_2(input)))
    print(encoder_1(input) - encoder_2(input))
    # equal是判断两个tensor是否完全相等的函数，这个有可能是false，感觉是因为浮点数的原因
    print(torch.equal(encoder_1(input), encoder_2(input)))
