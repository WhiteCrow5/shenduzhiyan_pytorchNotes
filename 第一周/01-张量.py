import numpy as np
import torch

#张量：多维数组
'''
Variable是torch.autograd中的数据类型，主要用于封装Tensor，进行自动求导
data: 被包装的Tensor
grad: data的梯度
grad_fn: 创建Tensor的Function，是自动求导的关键
requires_grad: 指示是否需要梯度
is_leaf: 指示是否是叶子结点（张量）

Tensor：张量 0.4.0开始，Variable并入Tensor
同上
dtype: 张量的数据类型
shape: 张量的形状
device: 张量所在设备

'''

#一、张量的直接创建
#1.1 torch.tensor()直接创建
'''
torch.tensor(
    data,           数据
    dtype=None,
    device=None,
    requires_grad=False,
    pin_memory=False)
'''
# t = torch.tensor([[1,3,4],[1,3,5]], device='cuda')
# print(t)

#1.2 使用torch.from_numpy(ndarry)创建(共享内存)
# arr = np.array([[1,2,3],[4,5,6]])
# t = torch.from_numpy(arr)
# print(arr)
# print(t)
# arr[1][1] = 1
# print(t)

#二、依据数值创建张量 torch.zeros() torch.ones() torch.full(size, fill_value)fill_value自定义填充的值
#2.1 torch.zeros()创建全为0的张量
'''
torch.zeros(
    *size,                  张量的形状(3,3),(3,28,28)
    out = None,             输出的张量
    dtype = None,           
    layout = torch.strided, 内存中布局形式
    device = None,
    requires_grad = False)
'''
# t = torch.zeros((3,4),dtype=torch.double)
# print(t)

#2.2 torch_zeros_like() 依据输入的形状创建全零张量
'''
torch.zeros_like(
    input,             #数据类型为tensor类型
    dtype = None,
    layout = None,
    device = None,
    requires_grad = False)
'''
# arr = np.array([[1,3,4],[3,4,5]])
# t = torch.tensor(arr)
# t = torch.zeros_like(t)
# print(t)

# t_ones = torch.ones((4,5))
# print(t_ones)
# t_any = torch.full((5,4),9)
# print(t_any)
# t_ones_like = torch.ones_like(t_any)
# t_any_like = torch.full_like(t_ones,8)
# print(t_ones_like)
# print(t_any_like)

#2.7 torch.arange()   创建等差的一维张量左闭右开
#2.8 torch.linspace() 创建等差的一维张量左闭右闭
#2.9 torch.logspace() 创建对数均分的一维张量
'''
torch.arange(
    start = 0,          数列的起始值
    end,                数列的结束值
    arange中: step = 1,                      数列公差 
    linspace中和logspace中: steps = 50，     数列长度
    logspace中: base = 10.0                  对数的底，默认为10.0
    out = None,
    dtype = None,
    layout = torch.strided,
    device = None,
    requires_grad = False)
'''
# t1 = torch.arange(50, 100, 2)
# t2 = torch.linspace(50, 100, 50)
# t3 = torch.logspace(1,10,10)
# print(t1)
# print(t2)
# print(t3)

#2.10 torch.eye()     创建单位对角矩阵（二维张量，默认为方阵）
'''
torch.eye(
    n,              矩阵行数
    m = None,       矩阵列数
    out = None,
    dtype = None, ...)
'''
# t = torch.eye(5)
# print(t)

#三、 依概率分布创建张量
#3.1 torch.normal() 生成正态分布(高斯分布)
'''
torch.normal(
    mean,       均值（标量和张量均可）
    std,        标准差（标量和张量均可）  但是必有一个张量
    out = None)
'''
# mean = torch.arange(1, 10 ,dtype = torch.float)
# std = torch.arange(1, 10 ,dtype = torch.float)
# mean = 1
# t = torch.normal(mean, std)
# print(t)

#3.2 torch.randn()          生成标准正态分布
#3.3 torch.randn_like()     生成标准正态分布
#3.4 torch.rand()           均匀分布
#3.5 torch.rand_like()
'''
torch.randn(
    *size,
    out = None...)
'''
# t1 = torch.randn((3,3))
# t2 = torch.rand((3,3))
# print(t1)
# print(t2)

#3.6 torch.randint() 生成区间[low, high)的整数均匀分布
#3.7 torch.randint_like() 生成区间[low, high)的整数均匀分布
'''
torch.randint(
    low = 0,
    high,
    size,...)
'''

#3.8 torch.randperm() 生成从0到n-1的随机排列
'''
torch.randperm(
    n,      张量的长度
    out = None,
    dtype = torch.int64...)
'''
# t = torch.randperm(6)
# print(t)

#3.9 torch.bernoulli() 以input为概率，生成伯努利分布（0-1分布）
'''
torch.bernoulli(
    input,
    *,
    generator = None,
    out = None)
'''