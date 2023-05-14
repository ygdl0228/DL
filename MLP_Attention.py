# @Time    : 2023/5/14 22:06
# @Author  : ygd
# @FileName: MLP_Attention.py
# @Software: PyCharm


from torch import nn
import torch
import matplotlib.pyplot as plt
import time


class MLP_Attention(nn.Module):
    def __init__(self, channel, reduction):
        super(MLP_Attention, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.ave_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_layer = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel)
        )

    def forward(self, x):
        a, b = x.size()
        y = x.view(a, 1, b)
        y = self.ave_pool(y).view(-1)
        y = self.fc_layer(y).view(a, 1)
        return (y * x).view(-1)


class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.attention = MLP_Attention(self.input_size, 1)
        self.fc_layer = nn.Sequential(
            nn.Linear(self.input_size * 6, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_size)
        )

    def forward(self, x):
        x = self.attention(x)
        y = x.view(-1)
        return self.fc_layer(x), self.fc_layer(y)


attention = MLP_Attention(4, 1)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
x1 = torch.rand(4, 6).to(device)
model1 = Net(4, 1).to(device)
model2 = Net(4, 1).to(device)
target = torch.Tensor([0])
optimizer1 = torch.optim.Adam(model1.parameters())
optimizer2 = torch.optim.Adam(model2.parameters())
ctitic = nn.SmoothL1Loss()
iter_max = 100
loss1_list = []
loss2_list = []
for i in range(iter_max):
    output1, _ = model1(x1)
    _, output2 = model2(x1)
    loss1 = ctitic(output1, target)
    loss2 = ctitic(output2, target)
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    loss1.backward()
    loss2.backward()
    optimizer1.step()
    optimizer2.step()
    loss1_list.append(loss1.item())
    loss2_list.append(loss2.item())
plt.plot(loss1_list,label='1')
plt.plot(loss2_list,label='2')
plt.legend()
plt.show()