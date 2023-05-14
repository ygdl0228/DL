# @Time    : 2023/5/14 15:34
# @Author  : ygd
# @FileName: mnist.py
# @Software: PyCharm

# 手写数字

from torchvision import datasets
from torchvision import transforms
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

batch_size = 512
train_loader = DataLoader(datasets.MNIST('mnist_data', train=True, download=False, transform=transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=batch_size, shuffle=True)

test_loader = DataLoader(datasets.MNIST('mnist_data/', train=False, download=False, transform=transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=batch_size, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc_layers = nn.Sequential(nn.Linear(28 * 28, 256),
                                       nn.BatchNorm1d(256),
                                       nn.ReLU(),
                                       nn.Linear(256, 64),
                                       nn.BatchNorm1d(64),
                                       nn.ReLU(),
                                       nn.Linear(64, 10)
                                       )

    def forward(self, x):
        return self.fc_layers(x)


def one_hot(label, depth=10):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    return out


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_list = []
loss = nn.MSELoss().to(device)
iter_max = 3
T1 = time.time()
for i in range(iter_max):
    for _, (x, y) in enumerate(train_loader):
        x = x.view(x.size(0), 28 * 28).to(device)
        output = model(x)
        y_onehot = one_hot(y).to(device)
        l = loss(output, y_onehot)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        loss_list.append(l.item())
T2 = time.time()
print(T2 - T1)
plt.plot(loss_list)
plt.show()
