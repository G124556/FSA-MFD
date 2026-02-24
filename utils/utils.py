"""
工具函数: 日志记录、可视化、模型保存/加载
"""

import os
import sys
import logging
import shutil
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------

def setup_logger(log_dir: str, name: str = 'fsamfd') -> logging.Logger:
    """
    配置 Logger: 同时输出到终端和文件
    
    Args:
        log_dir: 日志保存目录
        name:    Logger 名称
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        '[%(asctime)s][%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 终端 handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    # 文件 handler
    log_file = os.path.join(log_dir, f'{name}_{time.strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Checkpoint 管理
# ---------------------------------------------------------------------------

class CheckpointManager:
    """模型 Checkpoint 保存和加载管理器"""

    def __init__(self, save_dir: str, max_keep: int = 5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self.best_f1 = 0.0
        self.best_epoch = 0
        self._checkpoint_queue = []

    def save(self, model: nn.Module, optimizer, scheduler,
             epoch: int, metrics: Dict[str, float],
             is_best: bool = False) -> str:
        """
        保存 checkpoint
        
        Args:
            model:     模型
            optimizer: 优化器
            scheduler: 学习率调度器
            epoch:     当前 epoch
            metrics:   评估指标字典
            is_best:   是否为最佳模型
        Returns:
            保存路径
        """
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
        }

        # 保存当前 checkpoint
        ckpt_path = self.save_dir / f'checkpoint_epoch_{epoch:03d}.pth'
        torch.save(state, ckpt_path)

        # 管理 checkpoint 数量
        self._checkpoint_queue.append(ckpt_path)
        if len(self._checkpoint_queue) > self.max_keep:
            old_ckpt = self._checkpoint_queue.pop(0)
            if old_ckpt.exists() and 'best' not in str(old_ckpt):
                old_ckpt.unlink()

        # 保存最佳模型
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            shutil.copy(ckpt_path, best_path)
            self.best_f1 = metrics.get('F1', 0.0)
            self.best_epoch = epoch

        # 始终保存最新
        latest_path = self.save_dir / 'latest.pth'
        shutil.copy(ckpt_path, latest_path)

        return str(ckpt_path)

    def load(self, model: nn.Module, path: str,
             optimizer=None, scheduler=None,
             device: str = 'cpu') -> Dict[str, Any]:
        """
        加载 checkpoint
        
        Returns:
            包含 epoch 和 metrics 的字典
        """
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
        }


# ---------------------------------------------------------------------------
# 可视化工具
# ---------------------------------------------------------------------------

def save_change_map(pred: torch.Tensor, save_path: str,
                    threshold: float = 0.5):
    """
    保存变化预测图
    
    Args:
        pred:      预测概率 [1, H, W] 或 [H, W]
        save_path: 保存路径
        threshold: 二值化阈值
    """
    if pred.dim() == 3:
        pred = pred.squeeze(0)

    pred_np = pred.cpu().numpy()
    binary = (pred_np > threshold).astype(np.uint8) * 255
    Image.fromarray(binary).save(save_path)


def visualize_comparison(img_t: torch.Tensor, img_t1: torch.Tensor,
                          pred: torch.Tensor, label: torch.Tensor,
                          save_path: str, mean=None, std=None):
    """
    保存对比可视化图 (时相1 | 时相2 | 预测 | 真值)
    
    Args:
        img_t:     时相1图像 [3, H, W]
        img_t1:    时相2图像 [3, H, W]
        pred:      预测概率 [1, H, W]
        label:     真实标签 [H, W]
        save_path: 保存路径
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    def denormalize(tensor):
        """逆归一化到 [0, 255]"""
        t = tensor.clone().cpu()
        for i, (m, s) in enumerate(zip(mean, std)):
            t[i] = t[i] * s + m
        t = (t * 255).clamp(0, 255).byte()
        return t.permute(1, 2, 0).numpy()

    img_t_vis = denormalize(img_t)
    img_t1_vis = denormalize(img_t1)

    H, W = img_t_vis.shape[:2]

    pred_np = pred.squeeze(0).cpu().numpy()
    pred_binary = (pred_np > 0.5).astype(np.uint8) * 255
    pred_vis = np.stack([pred_binary] * 3, axis=-1)

    label_np = label.cpu().numpy().astype(np.uint8) * 255
    label_vis = np.stack([label_np] * 3, axis=-1)

    # 横向拼接
    combined = np.concatenate([img_t_vis, img_t1_vis, pred_vis, label_vis], axis=1)
    Image.fromarray(combined).save(save_path)


# ---------------------------------------------------------------------------
# 其他工具
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'total_M': round(total / 1e6, 2),
        'trainable_M': round(trainable / 1e6, 2),
    }


def set_random_seed(seed: int = 42):
    """设置随机种子保证可复现"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 20, min_delta: float = 0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.should_stop
