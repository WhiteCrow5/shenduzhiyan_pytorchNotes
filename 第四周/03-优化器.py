'''
class Optimizer(obj):
    def __init__(self,params,defaults):
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.param_groups = []
        ...
        param_groups = []{'params': param_groups}
'''

#1. optim.SGD
'''
optim.SGD(
    params, 管理参数值
    lr = <obj obj>,
    momentum = 0,
    dampening = 0,
    weight_decay = 0,   L2正则化系数
    nesterov = Fasle    是否采用NAG
    )
'''
'''
1. optim.SGD            随机梯度下降法
2. optim.Adagrad        自适应学习率梯度下降法
3. optim.RMSprop        Adagrad的改进
4. optim.Adadelta       Adagrad的改进
5. optim.Adam           RMSprop结合Momentum
6. optim.Adamax         Adam增加学习率上限
7. optim.SparseAdam     稀疏版的Adam
8. optim.ASGD           随机平均梯度下降
9. optim.Rprop          弹性反向传播
10. optim.LBFGS         BFGS的改进

'''