# from __future__ import print_function, division
import pickle
import sys

import torch
import numpy as np
# from sklearn.preprocessing import StandardScaler
import random
from PIL import Image
import torch.utils.data as data
import os
import os.path

from src.utils.logger import Logger


def imbalance(imgs):
    start = True
    for item in imgs:
        label = item[1].reshape(-1, item[1].shape[0])
        if start:
            lable_matrix = label
        else:
            lable_matrix = np.concatenate((lable_matrix, label), axis=0)
        start = False
    match_matrix = np.matmul(lable_matrix, lable_matrix.T)
    match_num = np.sum(match_matrix > 0)
    return match_num


def make_dataset(image_list, labels):
    if labels:  # labels=None for imagenet
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:  # split and get the labels
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    # from torchvision import get_image_backend
    # if get_image_backend() == 'accimage':
    #    return accimage_loader(path)
    # else:
    return pil_loader(path)


class ImageList(object):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        train_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        test_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, option, image_list, labels=None, train_transform=None, test_transform=None, strong_transform=None,Train=False,
                 loader=default_loader):  # ImageList(image_list = '../data/imagenet/train.txt')
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images" ))

        self.imgs = imgs
        self.option = option
        self.train_transform = train_transform
        self.strong_transform = strong_transform
        self.test_transform = test_transform
        self.loader = loader
        self.Train = Train

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # if self.option.data_name == 'cifar10':
        #     pilImg = Image.fromarray(self.data[index])
        #     return (self.transform(pilImg), self.labels[index])

        path, target = self.imgs[index]

        #####  here to adapte my path #####

        #########################
        img = self.loader(path)
        if min(img.size) < 224:
            # 自动调整小图像
            from torchvision.transforms import functional as F
            from torchvision.transforms import InterpolationMode

            width, height = img.size
            scale_factor = 224 / min(width, height) * 1.1
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img = F.resize(img, [new_height, new_width], interpolation=InterpolationMode.BILINEAR)

        # if self.train_transform is None or self.test_transform is None:
        #     Logger.info("\n\t=======transform is none=====\t\n")
        #     sys.exit(0)
        if self.Train:
            if self.train_transform is not None:
                img_i = self.train_transform(img)
                img_j = self.train_transform(img)
                image_origin = self.test_transform(img)
                return ((img_i, img_j, image_origin), target, index)
        else:
            image = self.test_transform(img)
            return (image, target)

    # if self.test_transform is not None:
    #     target = self.test_transform(target)
    # print("shape {} path {}".format(img.shape, path))

    def __len__(self):
        return len(self.imgs)
