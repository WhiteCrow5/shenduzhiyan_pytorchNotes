import torchvision
'''
Container
    nn.Sequential    按顺序包装多个网络层
    nn.ModuleList   同list一样包装网络层
    nn.MoudleDict   同dict一样包装网络层
'''
'''
    Sequential：顺序性，各网络层之间严格按顺序执行，常用于block构建
    ModuleList：迭代性，常用于大量重复网构建， 通过for循环
    MoudleDict：索引性，常用于可选择的网络
'''

torchvision.models.AlexNet()
torchvision.models.DenseNet()
torchvision.models.GoogLeNet()
torchvision.models.Inception3()
torchvision.models.MNASNet()
torchvision.models.MobileNetV2()
torchvision.models.ResNet()
torchvision.models.ShuffleNetV2()
torchvision.models.SqueezeNet()
torchvision.models.VGG()