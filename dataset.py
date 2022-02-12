from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader


class HotdogData(Dataset):
    def __init__(self, img_path, transforms=None):
        # 初始化，读取数据集
        self.transforms = transforms
        self.img_path = img_path
        self.pos_dir = img_path + '/hotdog'
        self.neg_dir = img_path + '/not-hotdog'
        self.pos_num = len(os.listdir(self.pos_dir))
        self.neg_num = len(os.listdir(self.neg_dir))

    def __len__(self):
        return self.pos_num + self.neg_num

    def __getitem__(self, index):
        if index < self.pos_num:  # 获取正样本
            label = 1
            img = Image.open(
                self.pos_dir + '/' + str(index if self.img_path[-5:] == 'train' else index + 1000) + '.png')
        else:  # 获取负样本
            label = 0
            img = Image.open(self.neg_dir + '/' + str(
                (index - self.pos_num) if self.img_path[-5:] == 'train' else index - self.pos_num + 1000) + '.png')

        if self.transforms:
            img = self.transforms(img)

        return img, label


def prepare_data(params):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),  # 将图像随意裁剪，宽高均为224
        transforms.RandomHorizontalFlip(),  # 以0.5的概率左右翻转图像
        transforms.RandomVerticalFlip(),
        #                     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
        transforms.RandomRotation(degrees=5, expand=False, fill=None),
        transforms.ToTensor(),  # 将PIL图像转为Tensor，并且进行归一化
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # 将PIL图像转为Tensor，并且进行归一化
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
    ])

    train_data = HotdogData(r'hotdog\train', transforms=train_transform)
    trainloader = DataLoader(train_data, batch_size=params.batch_size, shuffle=True)

    test_data = HotdogData(r'hotdog\test', transforms=test_transform)
    testloader = DataLoader(test_data, batch_size=params.batch_size, shuffle=True)
    return trainloader, testloader