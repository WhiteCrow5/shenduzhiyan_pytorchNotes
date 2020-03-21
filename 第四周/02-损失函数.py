import torch.nn as nn

nn.BCELoss()

#1. nn.CrossEntropyLoss() nn.LogSoftmax()与nn.NLLLoss()结合，
#   进行交叉熵计算
'''
nn.CrossEntropyLoss(
    weight = None,          各类别的loss设置权值
    size_average = None,
    ignore_index = -100,    忽略某个类别
    reduce = None,
    reduction = 'mean'      计算模式，可为none/sum/mean
    )
'''

#2. nn.NLLLoss()   实现负对数似然函数中的负号功能
'''
nn.NLLLoss(
    weight = None,          各类别的loss设置权值
    size_average = None,
    ignore_index = -100,    忽略某个类别
    reduce = None,
    reduction = 'mean'      计算模式，可为none/sum/mean
    )
'''

#3. nn.BCELoss()   二分类交叉熵， 输入值取值在[0,1]
'''
nn.BCELoss(
    weight = None,          各类别的loss设置权值
    size_average = None,    忽略某个类别
    reduce = None,
    reduction = 'mean'      计算模式，可为none/sum/mean
    )
'''

#4. nn.BCEWithLogitsLoss() 结合Sigmoid与二分类交叉熵,网络最后不加Sigmoid函数
'''
nn.BCELoss(
    weight = None,          各类别的loss设置权值
    size_average = None,    忽略某个类别
    reduce = None,
    reduction = 'mean'      计算模式，可为none/sum/mean
    pos_weight = None
    )
'''

#5. nn.L1Loss()  计算inputs与target之差的绝对值
'''
nn.L1Loss(size_average=None, reduce=None, reduction='mean')
'''
#6. nn.MSELoss() 计算inputs与target之差的平方
'''
nn.MSELoss(size_average=None, reduce=None, reduction='mean')
'''
#7. SmoothL1loss()    平滑的L1Loss
'''
nn.SmoothL1loss(size_average=None, reduce=None, reduction='mean')
'''
#8. PoissonNLLoss()   泊松分布的负对数似然损失函数

#9. nn.KLDivLoss()    KLD KL散度，相对熵

#10. nn.MarginRankingLoss()     计算两个向量之间的相似度，用于排序任务

#11. nn.MultiLabelMarginLoss()  多标签边界损失函数

#12. nn.SoftMarginLoss()    计算二分类的Logistic损失

#13. nn.MultiLabelSoftMarginLoss()  SoftMarginLoss的多标签版本

#14. nn.MultiMarginLoss()   计算多分类的折页损失

#15. nn.TripletMarginLoss() 计算三元组损失，人脸验证中常用

#16. nn.HingeEmbeddingLoss()    计算两个输入的相似性，常用于非线性embedding和半监督学习

#17. nn.CosineEmbeddingLoss()   采用余弦相似度计算两个输入的相似性

#18. nn.CTCLoss()   计算CTC损失，解决时序类数据的分类