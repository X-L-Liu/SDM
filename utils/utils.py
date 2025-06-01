from datetime import datetime
from typing import Callable, Optional
from torch import nn
from torchvision.datasets import VisionDataset
from torchvision import transforms
from .autoaugment import *
from .cutout import Cutout
import os
import pickle
import torch
import torchattacks

transform_cifar10_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4, fill=128),
    transforms.RandomHorizontalFlip(),
    CIFAR10Policy(),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=16)
])

transform_cifar10_train_resize64 = transforms.Compose([
    transforms.RandomCrop(32, padding=4, fill=128),
    transforms.RandomHorizontalFlip(),
    CIFAR10Policy(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=16),
])

transform_cifar10_test = transforms.Compose([
    transforms.ToTensor()
])

transform_cifar10_test_resize64 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

transform_cifar100_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4, fill=128),
    transforms.RandomHorizontalFlip(),
    CIFAR10Policy(),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=16)
])

transform_cifar100_train_resize64 = transforms.Compose([
    transforms.RandomCrop(32, padding=4, fill=128),
    transforms.RandomHorizontalFlip(),
    CIFAR10Policy(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=16)
])

transform_cifar100_test = transforms.Compose([
    transforms.ToTensor()
])

transform_cifar100_test_resize64 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

transform_svhn_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4, fill=128),
    transforms.RandomHorizontalFlip(),
    CIFAR10Policy(),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=16)
])

transform_svhn_train_resize64 = transforms.Compose([
    transforms.RandomCrop(32, padding=4, fill=128),
    transforms.RandomHorizontalFlip(),
    CIFAR10Policy(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=16)
])

transform_svhn_test = transforms.Compose([
    transforms.ToTensor()
])

transform_svhn_test_resize64 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

transform_miniimagenet_train = transforms.Compose([
    transforms.RandomCrop(64, padding=8, fill=128),
    transforms.RandomHorizontalFlip(),
    CIFAR10Policy(),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=24),
])

transform_miniimagenet_test = transforms.Compose([
    transforms.ToTensor()
])


class MiniImageNet(VisionDataset):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 pixel: str = '64',
                 transform: Optional[Callable] = None
                 ):
        super().__init__(root, transform=transform)
        data = pickle.load(file=open(os.path.join(self.root, 'mini-imagenet-' + pixel + '.pkl'), 'rb'))
        if train:
            self.sample, self.label = data['train_sample'], data['train_label']
        else:
            self.sample, self.label = data['test_sample'], data['test_label']
        self.sample = self.sample.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index: int):
        img, target = self.sample[index], self.label[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self) -> int:
        return len(self.label)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.num = 0
        self.total_num = 0
        self.rate = 0

    def update(self, num, total_num):
        self.num += num
        self.total_num += total_num
        self.rate = self.num / self.total_num


def seed_torch(seed=2025):  # 随机数种子
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
