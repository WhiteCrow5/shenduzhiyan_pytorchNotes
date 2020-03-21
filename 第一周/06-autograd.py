import torch

#1.1 torch.autograd.backward()  自动求取梯度
'''
torch.autograd.back(
    tensors,                用于求导的张量， 如loss
    grad_tensors = None,    保存计算图
    retain_graph = None,    创建导数计算图，用于高阶求导
    create_graph = False)   多梯度权重
'''
# t1 = torch.tensor([1.], requires_grad=True)
# t2 = torch.tensor([2.], requires_grad=True)
#
# a = 2 * t1 + 2
# b = t2 * t1 + 3
#
# y1 = a * b
# y2 = a + b
# #将retrain_graph设置为ture后，可以再次使用backward
# y2.backward(retain_graph=True)
# print(t1.grad)
# print(t2.grad)
# #梯度不会自动清零
# t1.grad.zero_()
# y2.backward()
# print(t1.grad)
# print(t2.grad)


# loss = torch.cat([y1,y2], dim=0)
# grad_tensors = torch.tensor([1.,2.])
# loss.backward(gradient=grad_tensors)
# print(t1.grad)
# print(t2.grad)

#1.2 torch.autograd.grad()  求取梯度
'''
torch.autograd.grad(
    outputs,                用于求导的张量
    inputs,                 需要梯度的张量
    grad_outputs = None,    创建导数 计算图，用于高阶求导
    retain_graph = None,    保存计算图
    create_graph = False)   多维度权重
'''
# x = torch.tensor([3.],requires_grad=True)
# y = torch.pow(x,2)
#
# grad_1 = torch.autograd.grad(y, x, create_graph=True)
# print(grad_1)
# print(grad_1[0])
# grad_2 = torch.autograd.grad(grad_1[0], x) #grad_1类型时元组类型
# print(grad_2)

#--------注意事项---------
'''
1.梯度不自动清零   torch.grad.zero_()
2.依赖于叶子结点的结点，requires_grad默认为True
3.叶子结点不可执行in-place操作
'''