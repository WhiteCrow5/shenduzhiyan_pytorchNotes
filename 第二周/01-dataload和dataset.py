from PIL import Image
#1. torch.utils.data.DataLoader()   构建可迭代的数据装载器
'''
DataLoader(
    dataset,                Dataseta类，决定数据从哪读取和如何读取
    batch_size = 1,         批大小
    shuffle = False,        每个epoch是否乱序
    sampler = None,
    batch_sampler = None,
    num_workers = 0,        是否多进程读取
    collate_fn = None,
    pin_memory = False,
    drop_last = False,      当样本数不能被batchsize整除时，是否舍弃最后一批数据
    timeout = 0,
    worker_init_fn = None,
    mutilprocessing_context = None)
'''

#2. torch.utils.data.Dataset()  Dataset抽象类，所有自定义的Dataset需要继承它，并且复写__getitem__()
'''
class Dataset(object):
    def __getitem__(self, index):   接受一个索引，返回一个样本
        raise NotImplementedError
    def __add__(self, other):
        return ConcatDataset([self, other])
'''

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
#from model.lenet import LeNet
#from tools.my_dataset import RMBDataset

class ImgDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.label_name = {'0':0,'1':1,'2':2,'3':3}
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()

