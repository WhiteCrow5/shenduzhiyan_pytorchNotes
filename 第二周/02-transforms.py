'''
torchvision.transforms:常用的图像预处理方法
    数据中心化、数据标准化、缩放、裁剪、旋转、翻转、填充
    噪声添加、灰度变换、线性变换、仿射变换、亮度变换
    饱和度变换、对比度变换
torchvision.datasets:常用数据集的dataset实现, MNIST, CIFAR-10, ImageNet
torchvision.model:常用的模型预训练, AlexNet, VGG, ResNet, GoogLeNet
'''

import torch
import torchvision.transforms as transforms
from PIL import Image
'''
transforms.Normalize(
    mean,
    std,
    inplace = False)
'''
#transforms.Crop
'''
transforms.CenterCrop(
    size)
'''
image = Image.open(r'RMB_data/1/0D6HCAXL.jpg').convert('RGB')
#transforms.CenterCrop((500,256))(image).show()

'''
transfroms.RandomCrop(
    size,   所需裁剪图片尺寸
    padding = None, 设置填充大小 (a,b,c,d) 左上右下分别填充a,b,c,d个像素
    pad_if_needed = False,
    fill = 0,
    padding_mode = 'constant'   填充模式
    )
'''
# for i in range(3):
#     transforms.RandomCrop(250)(image).show()

'''
transform.RandomResizedCrop(
    size,   所需裁剪图片尺寸
    scale = (0.08, 1),  随机裁剪面积比例，默认(0.08,1)
    ratio = (3/4, 4/3),  随机长宽比，默认(3/4,4/3)
    interpolation   插值方法PIL.Image.NEAREST BILINEAR BICUBIC
    )
'''
# transforms.RandomResizedCrop(250)(image).show()
'''
在图像的上下左右及中心裁剪出尺寸为size的五张图片
transform.FiveCrop(size)
如上，并对五张图片进行水平或垂直镜像，获得十张图片
transfrom.TenCrop(
    size, 
    vertical_flip   是否垂直翻转
    )
'''
# images = transforms.FiveCrop(250)(image)
# for image in images:
#     image.show()

#2.transforms-Flip
'''
transforms.RandomHorizontalFlip(p = 0.5)
transforms.RandomVerticalFlip(p = 0.5)
p:翻转概率
'''

#3.transforms-Rotation
'''
transforms.RandomRotation(
    degrees,    旋转角度
    resample,   重采样方法
    expand,     是否扩大图片
    center      
    )
'''

#4.Pad
'''
transforms.Pad(
    padding,    设置填充大小
    fill = 0,   设置填充的像素值
    padding_mode    设置填充模式
    )
'''

#5.ColorJitter 调整亮度、对比度、饱和度和色相
'''
transform.ColorJitter(
    brightness = 0,     亮度调整因子  
                        为a，从[max(0,1-a),1+a]中随机选取，为(a,b)，从[a,b]中随机选取
    contrast = 0,       对比度参数   同上
    saturation = 0,     饱和度参数   同上
    hue = 0             色相参数     
                        为a， 从[-a, a] 0<= a <= 0.5
                        为(a, b) 从[a, b] -0.5<= a <= b <= 0.5
)
'''

#transforms.ColorJitter(hue=0.5)(image).show()

#6. Grayscale
#7. RandomGrayscale
'''
transforms.RandomGrayscale(
    num_output_channels,    输出通道数 只能是1或3
    p = 0.1     概率值，默认为0.1
    )
transforms.Grayscale(
    num_output-channels)
'''

#8.RandomAffine 仿射变换：旋转、平移、缩放、错切和翻转
'''
transforms.RandomAffine(
    degrees,        角度
    translate = None,   平移区间
    scale = None,   缩放比例
    shear = None,   错切角度设置
    resample = False,   重采样方式
    fillcolor = 0       填充颜色
    )
'''

#9.RandomErasing 对图片进行随机遮挡
'''
transform.RandomErasing(
    p = 0.5,    遮挡概率
    scale = (0.02, 0.03),   遮挡区域面积
    ratio = (0.3, 0.3),     遮挡区域长宽比
    value = 0,              遮挡区域像素值
    inplace = False)
'''
#10. transforms.Lambda 用户自定义方法
'''
transforms.Lambda(lambd)
'''
#11. transforms.Resize() 重新设定尺寸
#12. transforms.Normalize(mean, std, inplace)   标准化

#transforms操作
#1. transforms.RandomChoice 从一系列的transforms中随机挑选一个
#2. transforms.RandomApply 依据概率执行一组transforms
#3. transforms.RandomOrder  对一组transforms操作打乱顺序
'''
transforms.RandomChoice([transforms1, transforms2, ...])
transforms.RandomApply([transforms1, transforms2, ...], p=0.5)
transforms.RandomOrder([transforms1, transforms2, ...])
'''

#自定义transforms方法
'''
class Compose(obj):
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return
        
class MyTransforms(obj):
    def __init__(self,..):
        ...
    def __call__(self,img):
        ...
'''