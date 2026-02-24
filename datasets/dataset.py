"""
变化检测数据集加载器
支持: LEVIR-CD, WHU-CD, CDD (DSIFN-CD)

目录结构 (统一格式):
    dataset_root/
    ├── train/
    │   ├── A/        # 时相1图像
    │   ├── B/        # 时相2图像
    │   └── label/    # 变化标签 (0=不变, 255=变化)
    ├── val/
    │   ├── A/
    │   ├── B/
    │   └── label/
    └── test/
        ├── A/
        ├── B/
        └── label/
"""

import os
import random
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# 数据增强
# ---------------------------------------------------------------------------

class ChangeDetectionAugmentation:
    """双时相图像的联动数据增强"""

    def __init__(self, img_size: int = 256, is_train: bool = True):
        self.img_size = img_size
        self.is_train = is_train

        # ImageNet 归一化参数
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, img_t: Image.Image, img_t1: Image.Image,
                 label: Image.Image) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            img_t:  时相1 PIL 图像
            img_t1: 时相2 PIL 图像
            label:  标签 PIL 图像
        Returns:
            img_t_tensor, img_t1_tensor, label_tensor
        """
        # 统一尺寸
        img_t = img_t.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_t1 = img_t1.resize((self.img_size, self.img_size), Image.BILINEAR)
        label = label.resize((self.img_size, self.img_size), Image.NEAREST)

        if self.is_train:
            # 随机水平翻转
            if random.random() > 0.5:
                img_t = TF.hflip(img_t)
                img_t1 = TF.hflip(img_t1)
                label = TF.hflip(label)

            # 随机垂直翻转
            if random.random() > 0.5:
                img_t = TF.vflip(img_t)
                img_t1 = TF.vflip(img_t1)
                label = TF.vflip(label)

            # 随机旋转 (90度倍数)
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                img_t = TF.rotate(img_t, angle)
                img_t1 = TF.rotate(img_t1, angle)
                label = TF.rotate(label, angle)

            # 随机裁剪
            if random.random() > 0.5:
                crop_size = int(self.img_size * random.uniform(0.7, 1.0))
                i, j, h, w = T.RandomCrop.get_params(
                    img_t, output_size=(crop_size, crop_size))
                img_t = TF.crop(img_t, i, j, h, w)
                img_t1 = TF.crop(img_t1, i, j, h, w)
                label = TF.crop(label, i, j, h, w)
                img_t = img_t.resize((self.img_size, self.img_size), Image.BILINEAR)
                img_t1 = img_t1.resize((self.img_size, self.img_size), Image.BILINEAR)
                label = label.resize((self.img_size, self.img_size), Image.NEAREST)

            # 颜色抖动 (仅作用于图像, 不改变标签)
            if random.random() > 0.5:
                color_jitter = T.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
                img_t = color_jitter(img_t)
                img_t1 = color_jitter(img_t1)

        # 转 Tensor 并归一化
        img_t_tensor = TF.to_tensor(img_t)
        img_t1_tensor = TF.to_tensor(img_t1)
        img_t_tensor = TF.normalize(img_t_tensor, self.mean, self.std)
        img_t1_tensor = TF.normalize(img_t1_tensor, self.mean, self.std)

        # 标签: 255 -> 1 (变化), 0 -> 0 (不变)
        label_np = np.array(label)
        label_tensor = torch.from_numpy((label_np > 128).astype(np.int64))

        return img_t_tensor, img_t1_tensor, label_tensor


# ---------------------------------------------------------------------------
# 基础数据集类
# ---------------------------------------------------------------------------

class ChangeDetectionDataset(Dataset):
    """
    通用变化检测数据集
    
    Args:
        root (str):      数据集根目录
        split (str):     划分方式 ('train', 'val', 'test')
        img_size (int):  图像尺寸
        suffix (str):    图像文件后缀
    """

    def __init__(self, root: str, split: str = 'train',
                 img_size: int = 256, suffix: str = '.png'):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.suffix = suffix

        self.img_t_dir = self.root / split / 'A'
        self.img_t1_dir = self.root / split / 'B'
        self.label_dir = self.root / split / 'label'

        # 获取图像文件列表
        self.file_list = self._get_file_list()

        is_train = (split == 'train')
        self.augmentation = ChangeDetectionAugmentation(img_size, is_train)

        print(f"[Dataset] {split}: {len(self.file_list)} samples loaded from {root}")

    def _get_file_list(self) -> List[str]:
        """获取数据集文件名列表"""
        if not self.img_t_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.img_t_dir}")

        files = sorted([
            f.stem for f in self.img_t_dir.iterdir()
            if f.suffix.lower() in ['.png', '.jpg', '.tif', '.tiff']
        ])
        return files

    def __len__(self) -> int:
        return len(self.file_list)

    def _load_image(self, directory: Path, name: str) -> Image.Image:
        """尝试多种后缀加载图像"""
        for ext in [self.suffix, '.png', '.jpg', '.tif', '.tiff']:
            path = directory / (name + ext)
            if path.exists():
                return Image.open(path).convert('RGB')
        raise FileNotFoundError(f"Image not found: {directory / name}")

    def _load_label(self, name: str) -> Image.Image:
        """加载标签图像"""
        for ext in ['.png', '.jpg', '.tif']:
            path = self.label_dir / (name + ext)
            if path.exists():
                return Image.open(path).convert('L')
        raise FileNotFoundError(f"Label not found: {self.label_dir / name}")

    def __getitem__(self, idx: int) -> dict:
        name = self.file_list[idx]

        img_t = self._load_image(self.img_t_dir, name)
        img_t1 = self._load_image(self.img_t1_dir, name)
        label = self._load_label(name)

        img_t_tensor, img_t1_tensor, label_tensor = self.augmentation(
            img_t, img_t1, label)

        return {
            'img_t': img_t_tensor,
            'img_t1': img_t1_tensor,
            'label': label_tensor,
            'name': name
        }


# ---------------------------------------------------------------------------
# 各数据集的专用类 (可针对性配置)
# ---------------------------------------------------------------------------

class LEVIRCDDataset(ChangeDetectionDataset):
    """
    LEVIR-CD 数据集
    637对建筑变化图像 (1024x1024像素)
    官方划分: 80%/10%/10% train/val/test
    下载: https://justchenhao.github.io/LEVIR/
    """

    def __init__(self, root: str, split: str = 'train', img_size: int = 256):
        super().__init__(root, split, img_size, suffix='.png')


class WHUCDDataset(ChangeDetectionDataset):
    """
    WHU-CD 数据集
    7620对航空图像 (256x256像素)
    官方划分: 80%/10%/10% train/val/test
    下载: https://gpcv.whu.edu.cn/data/building_dataset.html
    """

    def __init__(self, root: str, split: str = 'train', img_size: int = 256):
        super().__init__(root, split, img_size, suffix='.png')


class CDDDataset(ChangeDetectionDataset):
    """
    CDD (Change Detection Dataset) / DSIFN-CD 数据集
    15952对多类变化图像 (256x256像素)
    包含道路、建筑、水体变化场景
    划分: 14400/1360/192 train/val/test
    下载: https://ieee-dataport.org/documents/change-detection-dataset
    """

    def __init__(self, root: str, split: str = 'train', img_size: int = 256):
        super().__init__(root, split, img_size, suffix='.jpg')


# ---------------------------------------------------------------------------
# DataLoader 工厂函数
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    'levir': LEVIRCDDataset,
    'levir-cd': LEVIRCDDataset,
    'whu': WHUCDDataset,
    'whu-cd': WHUCDDataset,
    'cdd': CDDDataset,
    'dsifn': CDDDataset,
    'dsifn-cd': CDDDataset,
}


def build_dataloader(dataset_name: str, root: str, split: str,
                     img_size: int = 256, batch_size: int = 8,
                     num_workers: int = 4, pin_memory: bool = True) -> DataLoader:
    """
    构建 DataLoader
    
    Args:
        dataset_name: 数据集名称 ('levir', 'whu', 'cdd')
        root:         数据集根目录
        split:        划分 ('train', 'val', 'test')
        img_size:     图像尺寸
        batch_size:   批次大小
        num_workers:  数据加载线程数
        pin_memory:   是否锁页内存
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Available: {list(DATASET_REGISTRY.keys())}")

    dataset_cls = DATASET_REGISTRY[dataset_name]
    dataset = dataset_cls(root=root, split=split, img_size=img_size)

    is_train = (split == 'train')
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train,
    )
    return loader
