# @Time    : 2023/5/8 19:08
# @Author  : ygd
# @FileName: initialize_parameters.py
# @Software: PyCharm

import torch
from torch import nn
from torch.nn import init
import numpy as np


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()

        self.f1 = nn.Linear(4, 2)
        self.f2 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.f1(x)
        x = nn.ReLU(x)
        x = self.f2(x)
        return x


net = net()
'''
for name, param in net.named_parameters():
    if name == 'f1.weight':
        init.normal_(param, mean=0, std=1)
    if name == 'f1.bias':
        init.constant_(param, 0)
'''

for m in net.modules():
    if isinstance(m, nn.Linear):
        init.normal_(m.weight, mean=0, std=1)
        print(m)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)
