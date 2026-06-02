import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


# 1. 定义读取图像的数据集类
class TxtImageDataset(Dataset):
    def __init__(self, image_paths, img_size=224):
        """
        从图像路径列表加载图像的数据集
        :param image_paths: 图像绝对路径列表
        :param img_size: 图像Resize的尺寸
        """
        self.image_paths = image_paths
        # 仅做Resize和转Tensor，不做归一化（避免影响统计计算）
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            # 读取图像并转为RGB格式（确保三通道）
            img = Image.open(img_path).convert('RGB')
            return self.transform(img)
        except Exception as e:
            raise ValueError(f"图像读取失败: {img_path}，错误信息: {str(e)}")


# 2. 从train.txt提取图像路径
def get_image_paths_from_txt(txt_path):
    """
    从train.txt中提取所有训练图像的绝对路径
    :param txt_path: train.txt的绝对路径
    :return: 图像绝对路径列表
    """
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"train.txt不存在: {txt_path}")

    image_paths = []
    # 获取txt文件所在目录（用于拼接绝对路径）
    txt_dir = os.path.dirname(txt_path)

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f.readlines(), 1):
            line = line.strip()
            if not line:
                continue  # 跳过空行

            # 解析"路径 标签"格式（分割第一个空格）
            parts = line.split(' ', 1)  # 只分割一次，避免路径中含空格的情况
            if len(parts) != 2:
                raise ValueError(f"train.txt第{line_num}行格式错误，应为'路径 标签'，实际为: {line}")

            img_rel_path, _ = parts  # 只需要图像路径，忽略标签
            # 拼接绝对路径
            img_abs_path = os.path.join(txt_dir, img_rel_path)
            # 验证图像是否存在
            if not os.path.isfile(img_abs_path):
                raise FileNotFoundError(f"train.txt第{line_num}行对应的图像不存在: {img_abs_path}")

            image_paths.append(img_abs_path)

    print(f"从{txt_path}成功提取{len(image_paths)}张训练图像路径")
    return image_paths


# 3. 计算均值和标准差的主函数
def calculate_stats_from_txt(txt_path, batch_size=32, num_workers=0, img_size=224):
    """
    从train.txt计算训练集的均值和标准差
    :param txt_path: train.txt的绝对路径
    :param batch_size: 批次大小
    :param num_workers: 多进程数量（Windows建议设为0）
    :param img_size: 图像Resize尺寸（与训练时一致）
    :return: (mean_list, std_list) 均值和标准差列表
    """
    # 步骤1：从txt提取图像路径
    image_paths = get_image_paths_from_txt(txt_path)

    # 步骤2：创建数据集和数据加载器
    dataset = TxtImageDataset(image_paths, img_size=img_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # 统计计算无需打乱
        num_workers=num_workers,
        pin_memory=True  # 加速GPU处理
    )

    # 步骤3：计算均值和标准差
    mean = torch.zeros(3)  # RGB三通道均值
    std = torch.zeros(3)  # RGB三通道标准差
    total_images = 0  # 累计图像数量

    print(f"开始计算均值和标准差（共{len(image_paths)}张图像）...")
    for batch_idx, images in enumerate(dataloader):
        batch_size = images.size(0)
        # 展平空间维度：[batch, 3, H*W]
        images_flat = images.view(batch_size, 3, -1)
        # 累加每个通道的均值和标准差
        mean += images_flat.mean(dim=2).sum(dim=0)  # 按通道计算均值并累加
        std += images_flat.std(dim=2).sum(dim=0)  # 按通道计算标准差并累加
        total_images += batch_size

        # 打印进度（每10批次更新一次）
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
            print(f"已处理{batch_idx + 1}/{len(dataloader)}批次，共{total_images}张图像")

    # 计算最终均值和标准差（除以总图像数）
    mean = mean / total_images
    std = std / total_images

    # 格式化结果（保留4位小数）
    mean_list = [round(m.item(), 4) for m in mean]
    std_list = [round(s.item(), 4) for s in std]

    # 打印结果（醒目显示）
    print("\n" + "=" * 60)
    print(f"训练集RGB通道均值: {mean_list}")
    print(f"训练集RGB通道标准差: {std_list}")
    print("=" * 60)

    return mean_list, std_list


# 4. 直接运行入口
if __name__ == "__main__":
    # -------------------------- 只需修改这里的路径 --------------------------
    # train.txt的绝对路径（根据你的实际路径填写）
    TRAIN_TXT_PATH = "G:/Projects/HHCH-main/data/ucm/train.txt"
    # ----------------------------------------------------------------------

    # 计算统计量（Windows用户建议将num_workers设为0）
    calculate_stats_from_txt(
        txt_path=TRAIN_TXT_PATH,
        batch_size=32,  # 内存小可以改16或8
        num_workers=0,  # Windows系统改为0，Linux/Mac保持4
        img_size=224  # 和训练时的图像尺寸一致
    )
