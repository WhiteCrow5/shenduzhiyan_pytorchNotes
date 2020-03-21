'''
计算图：描述运算的有向无环图
计算图包含 结点（Node）和边（Edge）
结点：数据，如向量，矩阵，张量
边：  运算，如加、减、卷积
'''
import torch
w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

a = torch.add(w,x)
a.retain_grad() #保存a对y的梯度
b = torch.add(w,1)
y = torch.mul(a,b)

y.backward()
print(w.grad,x.grad,a.grad,b.grad,y.grad)