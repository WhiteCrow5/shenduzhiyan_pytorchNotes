import os
import random
from PIL import Image
from torch.utils.data import Dataset

random.seed(1)
rmb_label = {'1':0, '100':1}

class RMBDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.label_name = {'1':0, '100':1}
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
        data_info = []
        for root, dirs, _ in os.walk(data_dir):
            #遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = rmb_label[sub_dir]
                    data_info.append((path_img, int(label)))
        return data_info

if __name__ == '__main__':
    data_dir = r'../RMB_data'
    data_info = []
    for root, dirs, _ in os.walk(data_dir):
        #print(root)
        #print(dirs)
        # 遍历类别
        for sub_dir in dirs:
            img_names = os.listdir(os.path.join(root, sub_dir))
            img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
            #print(img_names)
            # 遍历图片
            for img_name in img_names:
                path_img = os.path.join(root, sub_dir, img_name)
                label = rmb_label[sub_dir]
                data_info.append((path_img, int(label)))
    #img = Image.open(data_info[0][0]).convert('RGB')
    #img.show()
    print(data_info)