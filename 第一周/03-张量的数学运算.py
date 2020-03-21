import torch

'''
一、加减乘除
torch.add()
torch.addcdiv()
torch.addcmul()
torch.sub()
torch.div()
torch.mul()

二、对数、指数、幂函数
torch.log(input, out=None)
torch.log10(input, out=None)
torch.log2(input, out=None)
torch.exp(input, out=None)
torch.pow()

三、三角函数
torch.abs(input, out=None)
torch.acos(input, out=None)
torch.cosh(input, out=None)
torch.cos(input, out=None)
torch.asin(input, out=None)
torch.atan(input, out=None)
torch.atan2(input, other, out=None)
'''

#1.1 torch.add() 逐元素计算 input + alpha*other
#1.1 torch.addcmul() 逐元素计算 input + value*tensor1*tensor2
#1.1 torch.addcdiv() 逐元素计算 input + alpha*ensor1/tensor2
'''
torch.add(
    input,
    alpha=1,
    other,
    out=None)
torch.addcmul(
    input,
    value=1,
    tensor1,
    tensor2,
    out=None)
'''
# t0 = torch.randn((3,3))
# t1 = torch.ones_like(t0)
# t2 = torch.full_like(t0,2)
# print(t0)
# print(torch.add(t0,t1))
# print(torch.addcmul(t0,2,t1,t2))
# print(torch.addcdiv(t0,2,t1,t2))
