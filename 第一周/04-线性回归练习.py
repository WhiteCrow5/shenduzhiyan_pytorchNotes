import torch

x = torch.randn(20,1) * 10
y = 8.5 * x + 6

w = torch.randn((1),requires_grad=True)
b = torch.zeros((1),requires_grad=True)

for i in range(5000):
    #前向传播
    wx = torch.mul(w,x)
    y_pred = torch.add(wx,b)

    #计算 loss
    loss = (0.5 * (y - y_pred) ** 2).mean()

    #反向传播
    loss.backward()

    #更新参数
    lr = 0.001
    w.data.sub_(lr * w.grad)
    b.data.sub_(lr * b.grad)
    w.grad.data.zero_()
    b.grad.data.zero_()

    if i%10 == 0:
        print('loss:{}\t,w:{}\t,b:{}'.format(loss,w.item(),b.item()))