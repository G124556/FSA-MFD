"""
变化检测评估指标
F1-score, IoU (Intersection over Union), Overall Accuracy (OA)
"""

import torch
import numpy as np
from typing import Dict, Tuple


class ChangeDetectionMetrics:
    """
    变化检测评估指标计算器
    
    支持批次累积, 最终调用 compute() 获取指标
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        """重置所有累计值"""
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.TN = 0
        self.total_pixels = 0

    @torch.no_grad()
    def update(self, pred: torch.Tensor, label: torch.Tensor):
        """
        更新累计值
        
        Args:
            pred:  预测概率图 [B, 1, H, W] 或 [B, H, W]
            label: 真实标签  [B, H, W] (0/1)
        """
        if pred.dim() == 4:
            pred = pred.squeeze(1)

        # 二值化预测
        pred_binary = (pred > self.threshold).long()
        label = label.long()

        self.TP += ((pred_binary == 1) & (label == 1)).sum().item()
        self.FP += ((pred_binary == 1) & (label == 0)).sum().item()
        self.FN += ((pred_binary == 0) & (label == 1)).sum().item()
        self.TN += ((pred_binary == 0) & (label == 0)).sum().item()
        self.total_pixels += label.numel()

    def compute(self) -> Dict[str, float]:
        """
        计算所有指标
        
        Returns:
            dict with keys: F1, IoU, OA, Precision, Recall
        """
        eps = 1e-8

        precision = self.TP / (self.TP + self.FP + eps)
        recall = self.TP / (self.TP + self.FN + eps)

        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = self.TP / (self.TP + self.FP + self.FN + eps)
        oa = (self.TP + self.TN) / (self.total_pixels + eps)

        return {
            'F1':        round(f1 * 100, 4),
            'IoU':       round(iou * 100, 4),
            'OA':        round(oa * 100, 4),
            'Precision': round(precision * 100, 4),
            'Recall':    round(recall * 100, 4),
        }

    def __repr__(self) -> str:
        metrics = self.compute()
        return (f"F1={metrics['F1']:.2f}% | "
                f"IoU={metrics['IoU']:.2f}% | "
                f"OA={metrics['OA']:.2f}%")


def compute_metrics_from_confusion_matrix(
        TP: int, FP: int, FN: int, TN: int) -> Dict[str, float]:
    """从混淆矩阵直接计算指标"""
    eps = 1e-8
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = TP / (TP + FP + FN + eps)
    oa = (TP + TN) / (TP + FP + FN + TN + eps)

    return {
        'F1':        round(f1 * 100, 4),
        'IoU':       round(iou * 100, 4),
        'OA':        round(oa * 100, 4),
        'Precision': round(precision * 100, 4),
        'Recall':    round(recall * 100, 4),
    }


class AverageMeter:
    """跟踪均值和方差 (用于 loss 统计)"""

    def __init__(self, name: str = ''):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
        self._values = []

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self._values.append(val)

    @property
    def std(self) -> float:
        if len(self._values) < 2:
            return 0.0
        return float(np.std(self._values))

    def __repr__(self) -> str:
        return f"{self.name}: {self.avg:.4f} (±{self.std:.4f})"
