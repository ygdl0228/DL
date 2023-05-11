# @Time    : 2023/5/8 14:15
# @Author  : ygd
# @FileName: Attention_CNN.py
# @Software: PyCharm

import torch
from torch import nn
from torchvision import datasets, transforms
import torchvision
import torch.utils.data as Data
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )
        self.F = nn.Flatten()
        self.weight = nn.Parameter(torch.randn(1, 400))
        self.attention = nn.Sequential(
            nn.Linear(400, 400),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.F(x)
        attention_out = nn.Sigmoid()(self.weight * self.attention(x))
        x = x * attention_out
        x = self.layer2(x)
        return x


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    train_data = torchvision.datasets.MNIST(  # torchvision中有这一数据集，可以直接下载
        root='./MNIST/',  # 下载后存放位置
        train=True,  # train如果设置为True，代表是训练集
        transform=torchvision.transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成
        # torch.FloatTensor (C x H x W), 训练的时候 [0.0,255.0] normalize 成 [0.0, 1.0] 区间
        download=True  # 是否下载；如果已经下载好，之后就不必下载
    )
    test_data = torchvision.datasets.MNIST(root='./MNIST/', train=False)
    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=64,
        shuffle=True
    )
    test_x = torch.unsqueeze(test_data.data, dim=1).float()[:2000] / 255.0
    # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
    test_y = test_data.targets[:2000]
    cnn = CNN().to(device)
    critic = torch.optim.Adam(cnn.parameters())
    loss = torch.nn.CrossEntropyLoss()
    loss_return = []
    for i in range(10):
        for x, y in train_loader:
            l = loss(cnn(x), y)
            critic.zero_grad()
            l.backward()
            critic.step()
        print(f"第{i + 1}次训练，损失为：{loss(cnn(test_x), test_y).detach().numpy()}")
        loss_return.append(loss(cnn(test_x), test_y).detach().numpy())
    plt.plot(loss_return)
    plt.show()
