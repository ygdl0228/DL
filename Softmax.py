# @Time    : 2023/4/21 17:16
# @Author  : ygd
# @FileName: Softmax.py
# @Software: PyCharm

import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    net = net().to(device)
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    loss = nn.CrossEntropyLoss()  # 交叉熵
    critic = torch.optim.Adam(net.parameters())
    loss_return = []
    for _ in range(100):
        for x, y in train_iter:
            l = loss(net(x), y)
            critic.zero_grad()
            l.backward()
            critic.step()
            loss_return.append(l.detach().numpy())
    plt.plot(loss_return)
