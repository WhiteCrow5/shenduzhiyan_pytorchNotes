'''
class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1)
    def get_lr(self):
        raise NotImplementedError

    optimizer:关联的优化器
    last_epoch:记录epoch数
    base_lrs:记录初始学习率
    step():更新下一个epoch的学习率
    get_lr():虚函数，计算下一个epoch的学习率
'''

# 1.StepLR
'''
lr_scheduler.StepLR(optimizer, step_size,gamma=0.1,last_epoch=-1
功能：等间隔调整学习率
主要参数：
    step_size:调整间隔数
    gamma:调整系数
调整方式：lr = lr * gamma
'''

# 2.mULTIsTEPlr
'''
lr_scheduler.MultiStepLR(optimizer,milestones,gamma=0.1,last_epoch=1)
功能：按给定间隔调整学习率
主要参数：
    moliestones:设定调整时刻数
    gamma:调整系数
调整方式： lr = lr * gamma
'''

# 3.ExponentialLR
'''
lr_scheduler.ExponentialLR(optimizer,gamma,last_epoch=-1)
功能：按指数衰减调整学习率
主要参数：
    gamma：指数的底
调整方式： lr = lr * gamma ** epoch
'''

# 4. CosineAnnealingLR
'''
lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0,last_epoch=-1))
功能：余弦周期调整学习率
主要参数：
    T_max:下降周期
    eta_min:学习率下限
'''

# 5. ReduceLRonPlateau
'''
lr_scheduler.ReduceLRonPlateau(
    optimizer,
    mode = 'min',
    factor = 0.1,
    patience = 10,
    verbose = False,
    threshold = 0.0001,
    threshold_mode = 'rel',
    cooldown = 0,
    min_lr = 0,
    eps = 1e-08
)
功能：监控指标，当指标不再变化则调整
主要参数：
    mode: min/max 两种模式
    factor:调整系数
    patience:耐心
    cooldown:冷却时间
    verbose:是否打印日志
    min_lr:学习率下限
    eps:学习率衰减最小值
'''

# 6. LambdaLR
'''
lr_scheduler.LambdaLR(optimizer,lr_lambda,last_epoch=-1)
功能：自定义调整策略
主要参数：
    lr_lambda: function or list
'''