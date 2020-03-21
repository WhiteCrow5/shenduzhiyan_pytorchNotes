'''
torch.nn
    nn.Parameter    张量子类，表示可学习参数，如weight,bias
    nn.Module       所有网络层基类，管理网络属性
    nn.functional   函数具体实现，如卷积、池化、激活函数等
    nn.init         参数初始化方法
self._parameters = OrderedDict()
self._buffers = OrderedDict()
self._backward_hooks = OrderedDict()
self._forward_hooks = OrderedDict()
self._forward_pre_hooks = OrderedDict()
self._state_dict_hooks = OrderedDict()
self._load_state_dict_pre_hooks = OrderedDict()
self._modules = OrderedDict()


'''