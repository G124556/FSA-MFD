"""
Dual-Memory Contrastive Learning (DMCL) Module
论文: Frequency-domain style alignment for change detection via multi-dimensional feature decoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from collections import deque


class LocalMemoryBank(nn.Module):
    """
    局部记忆库
    
    聚焦于批次内边缘像素的对比学习。
    使用边缘感知采样策略，确保对比学习集中在最难分类的边界区域。
    """

    def __init__(self, feat_dim: int, num_anchors: int = 32,
                 num_samples: int = 16, temperature: float = 0.07):
        """
        Args:
            feat_dim: 特征维度
            num_anchors: 每批次采样锚点数量
            num_samples: 每个锚点正/负样本数量 N
            temperature: InfoNCE 温度参数
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.num_anchors = num_anchors
        self.num_samples = num_samples
        self.temperature = temperature

    def _detect_edge_pixels(self, pred_mask: torch.Tensor) -> torch.Tensor:
        """
        使用 distanceTransform 检测预测结果中的边缘像素
        
        Args:
            pred_mask: [B, H, W] 预测掩码 (二值化后)
        Returns:
            edge_mask: [B, H, W] 边缘像素布尔掩码
        """
        B, H, W = pred_mask.shape
        edge_masks = []
        pred_np = pred_mask.cpu().numpy().astype(np.uint8)

        for b in range(B):
            mask = pred_np[b]
            # 距离变换检测边缘
            dist_fg = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            dist_bg = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
            # 边缘定义为距离较小的像素
            edge = ((dist_fg < 3) & (mask > 0)) | ((dist_bg < 3) & (mask == 0))
            edge_masks.append(torch.from_numpy(edge.astype(np.float32)))

        return torch.stack(edge_masks).bool().to(pred_mask.device)

    def forward(self, features: torch.Tensor, labels: torch.Tensor,
                pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 特征图 [B, C, H, W]
            labels:   变化标签 [B, H, W]
            pred:     预测掩码 [B, H, W]
        Returns:
            local_loss: 局部对比学习损失
        """
        B, C, H, W = features.shape

        # 将特征图展平为像素级
        feat_flat = features.permute(0, 2, 3, 1).reshape(-1, C)   # [B*H*W, C]
        label_flat = labels.reshape(-1)                             # [B*H*W]

        # 检测边缘像素
        pred_binary = (pred > 0.5).long()
        edge_mask = self._detect_edge_pixels(pred_binary)          # [B, H, W]
        edge_flat = edge_mask.reshape(-1)                          # [B*H*W]

        # 获取边缘像素索引
        edge_indices = edge_flat.nonzero(as_tuple=True)[0]

        if len(edge_indices) < 4:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # 随机采样锚点
        num_anchors = min(self.num_anchors, len(edge_indices))
        perm = torch.randperm(len(edge_indices))[:num_anchors]
        anchor_indices = edge_indices[perm]

        anchor_feats = feat_flat[anchor_indices]      # [num_anchors, C]
        anchor_labels = label_flat[anchor_indices]    # [num_anchors]

        loss_sum = torch.tensor(0.0, device=features.device)
        valid_count = 0

        for i in range(num_anchors):
            anchor = anchor_feats[i]                  # [C]
            anchor_label = anchor_labels[i]

            # 找正负样本
            pos_mask = (label_flat == anchor_label) & edge_flat
            neg_mask = (label_flat != anchor_label) & edge_flat

            pos_idx = pos_mask.nonzero(as_tuple=True)[0]
            neg_idx = neg_mask.nonzero(as_tuple=True)[0]

            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue

            # 采样正负样本
            num_pos = min(self.num_samples, len(pos_idx))
            num_neg = min(self.num_samples, len(neg_idx))
            pos_sample_idx = pos_idx[torch.randperm(len(pos_idx))[:num_pos]]
            neg_sample_idx = neg_idx[torch.randperm(len(neg_idx))[:num_neg]]

            pos_feats = feat_flat[pos_sample_idx]    # [num_pos, C]
            neg_feats = feat_flat[neg_sample_idx]    # [num_neg, C]

            # InfoNCE 损失
            anchor_norm = F.normalize(anchor.unsqueeze(0), dim=-1)
            pos_norm = F.normalize(pos_feats, dim=-1)
            neg_norm = F.normalize(neg_feats, dim=-1)

            pos_sim = torch.matmul(anchor_norm, pos_norm.T) / self.temperature
            neg_sim = torch.matmul(anchor_norm, neg_norm.T) / self.temperature

            s_pos = pos_sim.sum()
            logits = torch.cat([pos_sim.squeeze(0), neg_sim.squeeze(0)], dim=0)
            labels_infonce = torch.zeros(logits.shape[0], device=features.device)
            labels_infonce[:num_pos] = 1.0 / num_pos

            # 对比损失
            loss_i = -torch.log(
                torch.exp(s_pos) / (torch.exp(s_pos) + torch.exp(neg_sim).sum() + 1e-8)
            )
            loss_sum = loss_sum + loss_i
            valid_count += 1

        if valid_count == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        return loss_sum / valid_count


class GlobalMemoryBank(nn.Module):
    """
    全局记忆库
    
    使用 FIFO 策略维护历史图像的类原型向量，
    通过跨批次原型对比增强全局判别能力。
    """

    def __init__(self, feat_dim: int, num_classes: int = 2,
                 bank_size: int = 1000, num_pos_neg: int = 16,
                 temperature: float = 0.07):
        """
        Args:
            feat_dim:    特征维度
            num_classes: 类别数量 (变化检测默认2类)
            bank_size:   记忆库容量
            num_pos_neg: 正/负样本原型数量 N
            temperature: InfoNCE 温度参数
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.bank_size = bank_size
        self.num_pos_neg = num_pos_neg
        self.temperature = temperature

        # 使用 deque 实现 FIFO
        self.memory_banks = [deque(maxlen=bank_size) for _ in range(num_classes)]

        # 注册 buffer 以在 state_dict 中保存 (可选)
        self.register_buffer('_dummy', torch.zeros(1))

    @torch.no_grad()
    def update(self, features: torch.Tensor, labels: torch.Tensor):
        """
        更新全局记忆库
        
        Args:
            features: [B, C, H, W]
            labels:   [B, H, W]
        """
        B, C, H, W = features.shape
        feat_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
        label_flat = labels.reshape(-1)

        for b in range(B):
            start = b * H * W
            end = start + H * W
            b_feats = feat_flat[start:end]
            b_labels = label_flat[start:end]

            for cls in range(self.num_classes):
                cls_mask = (b_labels == cls)
                if cls_mask.sum() > 0:
                    prototype = b_feats[cls_mask].mean(dim=0)  # 类原型
                    self.memory_banks[cls].append(prototype.cpu())

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        全局原型对比学习
        
        Args:
            features: [B, C, H, W]
            labels:   [B, H, W]
        Returns:
            global_loss
        """
        # 检查记忆库是否有足够的样本
        min_bank_size = min(len(bank) for bank in self.memory_banks)
        if min_bank_size < self.num_pos_neg:
            # 先更新再返回零损失
            self.update(features, labels)
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        B, C, H, W = features.shape
        device = features.device
        feat_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
        label_flat = labels.reshape(-1)

        loss_sum = torch.tensor(0.0, device=device)
        valid_count = 0

        # 从当前批次中采样查询特征
        for cls in range(self.num_classes):
            cls_mask = (label_flat == cls)
            if cls_mask.sum() == 0:
                continue

            query_feats = feat_flat[cls_mask]
            if len(query_feats) > 32:
                idx = torch.randperm(len(query_feats))[:32]
                query_feats = query_feats[idx]

            query_feats = F.normalize(query_feats, dim=-1)  # [N_q, C]

            # 从记忆库采样正样本 (同类) 和负样本 (异类)
            pos_bank = list(self.memory_banks[cls])
            neg_class = 1 - cls  # 二分类
            neg_bank = list(self.memory_banks[neg_class])

            num_pos = min(self.num_pos_neg, len(pos_bank))
            num_neg = min(self.num_pos_neg, len(neg_bank))

            pos_idx = torch.randperm(len(pos_bank))[:num_pos]
            neg_idx = torch.randperm(len(neg_bank))[:num_neg]

            pos_protos = torch.stack([pos_bank[i] for i in pos_idx]).to(device)
            neg_protos = torch.stack([neg_bank[i] for i in neg_idx]).to(device)

            pos_protos = F.normalize(pos_protos, dim=-1)  # [num_pos, C]
            neg_protos = F.normalize(neg_protos, dim=-1)  # [num_neg, C]

            # InfoNCE
            pos_sim = torch.matmul(query_feats, pos_protos.T) / self.temperature  # [N_q, num_pos]
            neg_sim = torch.matmul(query_feats, neg_protos.T) / self.temperature  # [N_q, num_neg]

            s_pos = pos_sim.sum(dim=1, keepdim=True)   # [N_q, 1]
            s_neg = torch.logsumexp(neg_sim, dim=1, keepdim=True)  # [N_q, 1]

            loss_cls = (-s_pos + torch.log(torch.exp(s_pos) + torch.exp(s_neg) + 1e-8)).mean()
            loss_sum = loss_sum + loss_cls
            valid_count += 1

        # 更新记忆库
        self.update(features, labels)

        if valid_count == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return loss_sum / valid_count


class DualMemoryContrastiveLearning(nn.Module):
    """
    双记忆对比学习 (DMCL)
    
    结合局部记忆库 (边缘感知批次内对比) 和全局记忆库 (跨批次原型对比)，
    在边界和原型级别提供互补的判别监督。
    """

    def __init__(self, feat_dim: int, bank_size: int = 1000,
                 num_anchors: int = 32, num_samples: int = 16,
                 temperature: float = 0.07, global_weight: float = 0.5):
        super().__init__()
        self.global_weight = global_weight

        self.local_bank = LocalMemoryBank(
            feat_dim=feat_dim,
            num_anchors=num_anchors,
            num_samples=num_samples,
            temperature=temperature
        )
        self.global_bank = GlobalMemoryBank(
            feat_dim=feat_dim,
            num_classes=2,
            bank_size=bank_size,
            num_pos_neg=num_samples,
            temperature=temperature
        )

    def forward(self, features: torch.Tensor, labels: torch.Tensor,
                pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 解码器输出特征 [B, C, H, W]
            labels:   真实变化标签 [B, H, W] (0/1)
            pred:     预测变化图   [B, H, W] (0~1 概率)
        Returns:
            total_contrastive_loss
        """
        L_local = self.local_bank(features, labels, pred)
        L_global = self.global_bank(features, labels)

        return L_local + self.global_weight * L_global
