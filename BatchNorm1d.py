# @Time    : 2023/5/8 15:25
# @Author  : ygd
# @FileName: BatchNorm1d.py
# @Software: PyCharm

import torch
from torch import nn
from torch.optim import Adam
import matplotlib.pyplot as plt


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(16, 8),
            nn.LeakyReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 4),
            nn.LeakyReLU(),
            nn.BatchNorm1d(4),
            nn.Linear(4, 2)
        )

    def forward(self, x):
        x = self.layer1(x)
        return x


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
net = net().to(device)
target = torch.Tensor([[0, 0]])
input = x = torch.randn(2, 16)
loss = nn.SmoothL1Loss()
optimizer = Adam(net.parameters())
loss_list = []
for _ in range(1000):
    l = loss(net(input), target)
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
    loss_list.append(l.item())
plt.plot(loss_list)
plt.show()
