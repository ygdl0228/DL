# @Time    : 2023/4/23 20:42
# @Author  : ygd
# @FileName: test_1.py
# @Software: PyCharm

import torch
from torch import nn


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.x = nn.Linear(10, 2)
        self.x.weight = nn.Parameter(torch.zeros(10, 1))
        self.x.bias = nn.Parameter(torch.Tensor(2).fill_(2))

    def forward(self, x):
        return self.x(x)


#

x=torch.Tensor(5,1)
print(x)

