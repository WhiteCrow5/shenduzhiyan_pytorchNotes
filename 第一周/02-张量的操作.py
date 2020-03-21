import torch

#一、张量的拼接与切分
#1.1 torch.cat()    将张量按维度dim进行拼接
#1.1 torch.stack()  在新创建的维度dim上进行拼接
'''
torch.cat(
    tensors,    张量序列
    dim = 0,    要拼接的维度
    out = None)
'''
# t = torch.ones((2,3))
# t1 = torch.tensor([[1,1,1],[1,1,1]])
# t2 = torch.tensor([[2,2,2],[2,2,2]])
# # t0 = torch.cat([t, t], dim=0)   #保留行格式, [4, 3]
# # t1 = torch.cat([t, t], dim=1)   #保留列格式, [2, 6]
# t3 = torch.stack([t1, t2], dim=0)   #torch.Size([2, 2, 3])
# t4 = torch.stack([t1, t2], dim=1)   #torch.Size([2, 2, 3])
# t5 = torch.stack([t1, t2], dim=2)   #torch.Size([2, 3, 2])
# print(t3)
# print(t4)
# print(t5)

#1.3 torch.chunk()  将张量按维度dim进行平均切分， 返回张量列表
'''
torch.chunk(
    input,      要切分的张量
    chunks,     要切分的份数
    dim=0)      要切分的维度
'''
# t = torch.arange(1,11)
# t= torch.tensor([[1,2,3],[2,3,4]])
# print(torch.chunk(t, 3, 1))
# print(torch.chunk(t, 2, 0))

#1.4 torch.split()  将张量按维度dim进行切分 返回张量列表
'''
torch.split(
    tensor,     要切分的张量
    split_size_or_sections,  为int时，表示每一份的长度；为list时，按list元素切分
    dim=0)
'''
# t = torch.ones((6,6))
# print(torch.split(t,[1,2,3]))

#二、张量索引
#2.1 torch.index_select()  在维度dim上，按index索引数据  返回依index索引数据拼接的张量
'''
torch.index_select(
    input,      要索引的张量
    dim,        要索引的维度
    index,      要索引数据的序号
    out = None) 
'''
# t = torch.randint(0,9, size=(3,3))
# idx = torch.tensor([0,2])
# print(t)
# print(torch.index_select(t,1,index = idx))

#2.2 torch.masked_select()  按mask中的True进行索引 返回一维张量
'''
torch.masked_select(
    input,      要索引的张量
    mask,       与input同形状的布尔类型张量
    out=None)
'''
# t = torch.randint(0,9, size=(3,3))
# idx = torch.tensor([[True,False,True],
#                    [True,False,False],
#                    [False,False,True]])
#torch.ge(5) 大于等于5
#torch.gt(5) 大于5
#torch.le(5) 小于等于5
#torch.lt(5) 小于5
# idx = t.ge(5)
# print(t)
# print(torch.masked_select(t,idx))

#三、张量变换
#3.1 torch.reshape() 变换张量形状（数据在内存连续时，共享内存）
'''
torch.reshape(
    input,      要变换的张量
    shape)      形状（元组）
'''

# t = torch.arange(0,9)
# print(t.reshape(3,3)

#3.2 torch.transpose() 交换两个张量的维度
#3.3 torch.t()      转置二维张量
'''
torch.transpose(
    input,      要变换的张量
    dim0,       要交换的维度1
    dim1)       要交换的维度2
'''
# t = torch.randint(0,3,size=(3,3,4))
# print(t)
# print(torch.transpose(t, 0, 1))

#3.4 torch.squeeze()    压缩长度为1的维度
#3.5 torch.unsqueeze()  依据dim扩展维度
'''
torch.squeeze(
    input,      
    dim=None,   None移除所有长度为1的轴，指定维度后，当且仅当该轴长度为1时，可以被移除
    out=None)
'''

t = torch.rand((1,2,3,1))
print(torch.squeeze(t))
print(torch.squeeze(t,dim=0))
print(torch.squeeze(t,dim=1))