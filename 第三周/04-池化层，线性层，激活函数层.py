
#nn.MaxPool2d
#nn.AvgPool2d
'''
nn.MaxPool2d(
    kernel_size,            池化核尺寸
    stride = None,          步长
    padding = 0,            填充个数
    dilation = 1,           池化核间隔大小
    return_indices = False, 记录池化像素索引
    ceil_mode = False       尺寸向上取整
    )
'''

'''
nn.AvgPool2d(
    kernel_size,                池化核尺寸
    stride = None,              步长
    padding = 0,                填充个数
    dilation = 1,               池化核间隔大小
    ceil_mode = False           尺寸向上取整
    count_include_pad = True    填充值用于计算
    divisor_override = None     除法因子
    )
'''

#nn.MaxUnpool2d 对二维信号进行最大值池化上采样
'''
nn.MaxUnpool2d(
    kernel_size,
    stride = None,
    padding = 0)
forward(self,input,indices,output_size=None)
'''

#nn.Linear  线性层
'''
nn.Linear(in_features, out_features, bias)
y = xW^T + bias
'''

#nn.Sigmoid
'''
y = 1 / (1 + e^(-x))
y' = y * (1-y)
'''
#nn.tanh
'''
y = sinx / cosx = (e^x-e(-x))/(e^x+e(-x)) = 2/(1+e^(-2x)) + 1
y' = 1-y^2
'''

#nn.ReLU
'''
y = max(x,0)
y'  = 1, x>0
    = undefined, x = 0
    = 0, x<0
'''

#nn.LeakyReLU   negative_slope负半轴斜率
#nn.PReLU       init:可学习斜率
#nn.RReLU       lower:均匀分布下限 upper:均匀分布上限