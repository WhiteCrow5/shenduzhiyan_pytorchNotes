import torch
import torch.nn as nn
import numpy as np
torch.manual_seed(10)

#1.生成数据
sample_nums = 100
mean_value = 1.7
bias = 1
n_data = torch.ones(sample_nums,2)
#类别0 数据 shape(100, 2)
x0 = torch.normal(mean_value * n_data, 1) + bias
#类别0 标签 shape(100, 1)
y0 = torch.zeros(sample_nums)
#类别1 数据 shape(100, 2)
x1 = torch.normal(-mean_value * n_data, 1) + bias
#类别1 标签 shape(100, 1)
y1 = torch.zeros(sample_nums)
train_x = torch.cat((x0, x1), 0)
train_y = torch.cat((y0, y1), 0)

#2.选择模型
class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.features = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x) :
        x = self.features(x)
        x = self.sigmoid(x)
        return x

#实例化模型
lr_net = LR()

#3.选择损失函数
loss_fn = nn.BCELoss()

#4.选择优化器
lr = 0.01
optimizer = torch.optim.SGD(lr_net.parameters(), lr = lr, momentum=0.9)

#训练模型
for i in range(1000):
    #前向传播
    y_pred = lr_net(train_x)
    #loss
    loss = loss_fn(y_pred.squeeze(), train_y)
    #反向传播
    loss.backward()
    #更新参数
    optimizer.step()
    if i % 10 == 0:
        print('loss:{}'.format(loss))