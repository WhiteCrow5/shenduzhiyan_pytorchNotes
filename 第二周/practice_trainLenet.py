import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from model.lenet import LeNet, LeNet2
from tools.my_dataset import RMBDataset
import matplotlib.pyplot as plt


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

#设置随机数种子
set_seed(2)
rmb_label = {'1': 0, '100': 1}

#参数设置
MAX_EPOCH =10
BATCH_SIZE = 16
LR = 0.01
log_interval =10
val_interval = 1

#-----------step 1-----------
split_dir = os.path.join("rmb_split")
train_dir = os.path.join(split_dir, "train")
valid_dir = os.path.join(split_dir, "valid")

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomCrop(32,padding=4),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean,norm_std),
])

valid_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean,norm_std),
])

#构建MyDataset实例
train_data = RMBDataset(data_dir = train_dir, transform = train_transform)
valid_data = RMBDataset(valid_dir,valid_transform)

#print(train_data.__getitem__(95))
#transforms.ToPILImage()(train_data.__getitem__(95)[0]).show()

#构建DataLoader
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)

#建立模型
net = LeNet2(classes=2)
net.initialize_weights()

#损失函数
criterion = nn.CrossEntropyLoss()

#优化器
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

#训练模型
train_curve = list()
valid_curve = list()

for epoch in range(MAX_EPOCH):
    loss_mean = 0.
    correct = 0.
    total = 0.

    net.train()
    for i, (inputs, labels) in enumerate(train_loader):
        #前向传播
        #inputs, labels = data
        print(inputs.size(),i)
        outputs = net(inputs)
        #计算损失
        loss = criterion(outputs,labels)
        #梯度清零
        optimizer.zero_grad()
        #后向传播
        loss.backward()
        #更新权重
        optimizer.step()

        # 打印训练信息
        loss_mean += loss.item()
        print(loss.item())
        train_curve.append(loss.item())
        if (i+1) % log_interval == 0:
            loss_mean = loss_mean / log_interval
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f}".format(
                epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean))
            loss_mean = 0.
    #更新学习率
    scheduler.step()


    # validate the model
    if (epoch+1) % val_interval == 0:

        correct_val = 0.
        total_val = 0.
        loss_val = 0.
        net.eval()
        with torch.no_grad():
            for j, data in enumerate(valid_loader):
                inputs, labels = data
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).squeeze().sum().numpy()

                loss_val += loss.item()

            valid_curve.append(loss_val/valid_loader.__len__())
            print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, j+1, len(valid_loader), loss_val, correct_val / total_val))


train_x = range(len(train_curve))
train_y = train_curve

train_iters = len(train_loader)
valid_x = np.arange(1, len(valid_curve)+1) * train_iters*val_interval # 由于valid中记录的是epochloss，需要对记录点进行转换到iterations
valid_y = valid_curve

plt.plot(train_x, train_y, label='Train')
plt.plot(valid_x, valid_y, label='Valid')

plt.legend(loc='upper right')
plt.ylabel('loss value')
plt.xlabel('Iteration')
plt.show()

# ============================ inference ============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.join(BASE_DIR, "test_data")

test_data = RMBDataset(data_dir=test_dir, transform=valid_transform)
valid_loader = DataLoader(dataset=test_data, batch_size=1)

for i, data in enumerate(valid_loader):
    # forward
    inputs, labels = data
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)

    rmb = 1 if predicted.numpy()[0] == 0 else 100
    print("模型获得{}元".format(rmb))