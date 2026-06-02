import os
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
from torchvision.transforms import transforms
from .data_list import ImageList
from src.utils.gaussian_blur import GaussianBlur
from src.calculate_stats import calculate_stats_from_txt
###############################################################################
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
import numpy as np
import random
from PIL import Image, ImageFilter, ImageEnhance
os.environ["OPENBLAS_NUM_THREADS"] = "1"

class AdaptiveColorJitter:
    """根据图像亮度自适应调整颜色增强强度"""

    def __init__(self, dark_threshold=0.3, bright_threshold=0.7):
        self.dark_threshold = dark_threshold
        self.bright_threshold = bright_threshold

        # 定义不同亮度级别的增强参数
        self.dark_jitter = transforms.ColorJitter(
            brightness=0.2,  # 较弱的亮度变化
            contrast=0.2,  # 较弱的对比度变化
            saturation=0.1,  # 较弱的饱和度变化
            hue=0.02  # 较小的色调变化
        )
        self.normal_jitter = transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05
        )
        self.bright_jitter = transforms.ColorJitter(
            brightness=0.4,  # 较强的亮度变化
            contrast=0.4,  # 较强的对比度变化
            saturation=0.3,  # 较强的饱和度变化
            hue=0.08  # 较大的色调变化
        )

    def __call__(self, img):
        # 将PIL图像转换为numpy数组计算亮度
        img_array = np.array(img)

        # 计算归一化的平均亮度 (0-1范围)
        if img_array.dtype == np.uint8:
            brightness = np.mean(img_array) / 255.0
        else:
            brightness = np.mean(img_array)

        # 根据亮度选择增强强度
        if brightness < self.dark_threshold:
            return self.dark_jitter(img)
        elif brightness > self.bright_threshold:
            return self.bright_jitter(img)
        else:
            return self.normal_jitter(img)

    def __repr__(self):
        return f"AdaptiveColorJitter(dark_threshold={self.dark_threshold}, bright_threshold={self.bright_threshold})"


class GaussianBlur:
    """高斯模糊增强"""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class ChannelShuffle:
    """通道随机交换 - 针对多光谱数据"""

    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            # 将图像转换为numpy数组进行通道操作
            img_np = np.array(img)
            channels = img_np.shape[2] if len(img_np.shape) == 3 else 1

            if channels > 1:
                # 随机打乱通道顺序
                indices = list(range(channels))
                random.shuffle(indices)
                img_np = img_np[:, :, indices]
                img = Image.fromarray(img_np)
        return img


class RandomRotate90:
    """随机90度旋转 - 遥感图像常有的方向变化"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            angle = random.choice([0, 90, 180, 270])
            img = F.rotate(img, angle)
        return img


class RandomBrightnessContrast:
    """随机亮度和对比度调整 - 模拟不同光照条件"""

    def __init__(self, brightness=0.2, contrast=0.2, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            # 亮度调整
            enhancer = ImageEnhance.Brightness(img)
            factor = 1 + random.uniform(-self.brightness, self.brightness)
            img = enhancer.enhance(factor)
            # 对比度调整
            enhancer = ImageEnhance.Contrast(img)
            factor = 1 + random.uniform(-self.contrast, self.contrast)
            img = enhancer.enhance(factor)
        return img


class RandomCropWithScale:
    """修复版的随机裁剪，自动处理小尺寸图像"""

    def __init__(self, size, scale_range=(0.4, 1.0)):
        self.size = size
        self.scale_range = scale_range

    def __call__(self, img):
        width, height = img.size
        # 如果图像太小，先放大到至少大于目标尺寸
        if min(width, height) < self.size:
            scale_factor = self.size / min(width, height) * 1.1  # 放大到比目标尺寸稍大
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img = F.resize(img, (new_height, new_width), interpolation=InterpolationMode.BILINEAR)
            width, height = new_width, new_height
        # 随机选择缩放比例，但要确保缩放后仍然足够大
        min_scale = max(self.scale_range[0], self.size / min(width, height))
        scale = random.uniform(min_scale, self.scale_range[1])
        new_size = [int(dim * scale) for dim in (height, width)]
        # 调整大小
        img = F.resize(img, new_size, interpolation=InterpolationMode.BILINEAR)
        # 获取裁剪参数
        i, j, h, w = transforms.RandomCrop.get_params(
            img, output_size=(self.size, self.size))
        img = F.crop(img, i, j, h, w)
        return img


class RandomNoise:
    """添加随机噪声 - 模拟传感器噪声"""

    def __init__(self, noise_level=0.02, p=0.3):
        self.noise_level = noise_level
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img_np = np.array(img).astype(np.float32)
            noise = np.random.normal(0, self.noise_level * 255, img_np.shape)
            img_np = img_np + noise
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_np)
        return img


def getDataLoader(option):
    '''
    :param option:
    :return: train_loader,test_loader
    '''
    global test_dataset, train_dataset
    # dataset_mean, dataset_std = calculate_stats_from_txt(txt_path="D:\shenyuanyaun\HHCH-main\HHCH-main\data/nwpu/train.txt")
    # aid
    dataset_mean, dataset_std = [[0.398, 0.4092, 0.3684], [0.1465, 0.1331, 0.1282]]
    # ucm
    # dataset_mean, dataset_std = [[0.4865, 0.4918, 0.4526], [0.1682, 0.158, 0.1502]]
    # nwpu
    # dataset_mean, dataset_std =[[0.3683, 0.3811, 0.3438],[0.1404, 0.1304, 0.1267]]

    train_transforms = transforms.Compose([
        # 几何变换
        transforms.Resize((224, 224)),
        RandomCropWithScale(size=224, scale_range=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),  # 遥感图像常有垂直对称
        RandomRotate90(p=0.5),
        # 颜色变换 - 使用自适应版本
        AdaptiveColorJitter(),  # 替换原来的ColorJitter
        RandomBrightnessContrast(brightness=0.2, contrast=0.2, p=0.5),
        # 质量变换
        GaussianBlur(sigma=(0.1, 1.5)),
        RandomNoise(noise_level=0.02, p=0.2),
        transforms.RandomGrayscale(p=0.1),
        # 最终转换
        transforms.ToTensor(),
        transforms.Normalize(
            mean=dataset_mean, std=dataset_std
        )
    ])

    # 测试集增强 - 更简单的处理
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Resize(256),
        # 颜色变换 - 使用自适应版本
        AdaptiveColorJitter(),  # 替换原来的ColorJitter
        transforms.CenterCrop(224),  # 使用CenterCrop而不是RandomCrop
        transforms.ToTensor(),
        transforms.Normalize(
            mean=dataset_mean, std=dataset_std
        )
    ])

    if option.data_name == "ucm":
        test_list = 'D:/shenyuanyaun/HSSR/data/ucm/test.txt'
        train_list = 'D:/shenyuanyaun/HSSR/data/ucm/train.txt'
        train_dataset = ImageList(option, open(train_list).readlines(), train_transform=train_transforms, test_transform = test_transforms, Train=True)
        test_dataset = ImageList(option, open(test_list).readlines(), test_transform=test_transforms)
    elif option.data_name == "nwpu":
        test_list = 'D:/shenyuanyaun/HSSR/data/nwpu/test.txt'
        train_list = 'D:/shenyuanyaun/HSSR/data/nwpu/train.txt'
        train_dataset = ImageList(option, open(train_list).readlines(), train_transform=train_transforms, test_transform = test_transforms, Train=True)
        test_dataset = ImageList(option, open(test_list).readlines(), test_transform=test_transforms)
    elif option.data_name == "aid":
        test_list = 'D:/shenyuanyaun/HSSR/data/aid/test.txt'
        train_list = 'D:/shenyuanyaun/HSSR/data/aid/train.txt'
        train_dataset = ImageList(option, open(train_list).readlines(), train_transform=train_transforms, test_transform = test_transforms, Train=True)
        test_dataset = ImageList(option, open(test_list).readlines(), test_transform=test_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=option.batch_size, shuffle=False, num_workers = option.workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=option.batch_size, shuffle=False, num_workers = option.workers)
    return train_loader, test_loader

    pass
