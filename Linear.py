# @Time    : 2023/4/21 16:36
# @Author  : ygd
# @FileName: Linear.py
# @Software: PyCharm

import torch
from torch import nn
from d2l import torch as d2l
from torch.utils import data
import matplotlib.pyplot as plt
import time


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(2, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layer1(x)
        return x


def load_array(data_arrays, batch_size, is_train=True):  # @save
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = d2l.synthetic_data(true_w, true_b, 1000)
    batch_size = 10
    data_iter = load_array((features, labels), batch_size)
    net = net().to(device)
    loss = nn.SmoothL1Loss()
    critic = torch.optim.Adam(net.parameters())
    loss_return = []
    start = time.time()
    for _ in range(1000):
        for x, y in data_iter:
            l = loss(net(x), y)
            critic.zero_grad()
            l.backward()
            critic.step()
        loss_return.append(loss(net(features), labels).detach().numpy())
    end = time.time()
    print('程序运行时间：%.5f' % (end - start))
    plt.plot(loss_return)
    plt.show()
