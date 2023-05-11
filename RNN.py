# @Time    : 2023/5/7 20:37
# @Author  : ygd
# @FileName: RNN.py
# @Software: PyCharm

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


class net(nn.Module):
    def __init__(self, intput_size, hidden_size, num_layers, output_size):
        super(net, self).__init__()
        self.intput_size = intput_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.rnn = nn.RNN(self.intput_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, h0):
        x, h0 = self.rnn(x, h0)
        x = x.view(-1, self.hidden_size)
        x = self.fc(x)
        x = x.unsqueeze(dim=0)
        return x, h0


device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
intput_size = 1
hidden_size = 16
num_layers = 1
output_size = 1
batch_size = 1
num_time_steps = 50
model = net(intput_size, hidden_size, num_layers, output_size).to(device)
loss = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters())
h0 = torch.zeros(num_layers, batch_size, hidden_size)
loss_list = []
for iter in range(1000):
    start = np.random.randint(3, size=1)[0]
    time_steps = np.linspace(start, start + 10, num_time_steps)
    data = np.sin(time_steps)
    data = data.reshape(num_time_steps, 1)
    x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)
    y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)
    output, h0 = model(x, h0)
    h0 = h0.detach()
    l = loss(output, y)
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
    loss_list.append(l.item())
plt.plot(loss_list)
plt.show()
