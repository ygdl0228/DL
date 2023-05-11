# @Time    : 2023/5/9 21:39
# @Author  : ygd
# @FileName: SEAttention.py
# @Software: PyCharm

import torch
from torch import nn

'''
注意力机制
Squeeze-and-Excitation Networks
https://arxiv.org/abs/1709.01507
'''

class SEAttention(nn.Module):
    def __init__(self, channel, reduction):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_layer = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.BatchNorm1d(channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc_layer(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


if __name__ == "__main__":
    input = torch.randn(50, 512, 7, 7)
    se = SEAttention(512, 8)
    output = se(input)
    print(output.size())
