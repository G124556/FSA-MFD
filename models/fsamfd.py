"""
FSA-MFD: 完整网络架构
Frequency-domain Style Alignment + Multi-Dimensional Feature Decoding

编码器: ResNet50 (Siamese结构, 共享权重)
解码器: 逐层 FSA -> MDFD, 跳跃连接
损失:   CE + 0.1 * DMCL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from .fsa import FrequencyStyleAlignment
from .mdfd import MDFDBlock
from .dmcl import DualMemoryContrastiveLearning


# ---------------------------------------------------------------------------
# ResNet50 编码器 (提取多尺度特征)
# ---------------------------------------------------------------------------

class ResNet50Encoder(nn.Module):
    """ResNet50 编码器，提取4层多尺度特征"""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = tv_models.resnet50(
            weights=tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # 拆分 ResNet50 各阶段
        self.layer0 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )                          # -> [B, 64, H/4, W/4]
        self.layer1 = backbone.layer1  # -> [B, 256, H/4, W/4]
        self.layer2 = backbone.layer2  # -> [B, 512, H/8, W/8]
        self.layer3 = backbone.layer3  # -> [B, 1024, H/16, W/16]
        self.layer4 = backbone.layer4  # -> [B, 2048, H/32, W/32]

    def forward(self, x: torch.Tensor) -> list:
        """
        Returns:
            list of features: [layer1, layer2, layer3, layer4]
            channels: [256, 512, 1024, 2048]
        """
        x0 = self.layer0(x)
        x1 = self.layer1(x0)   # 256
        x2 = self.layer2(x1)   # 512
        x3 = self.layer3(x2)   # 1024
        x4 = self.layer4(x3)   # 2048
        return [x1, x2, x3, x4]


# ---------------------------------------------------------------------------
# FSA-MFD 主网络
# ---------------------------------------------------------------------------

class FSAMFD(nn.Module):
    """
    FSA-MFD: 基于频域风格对齐和多维特征解码的变化检测网络
    
    Args:
        input_size (int):       输入图像尺寸 (假设为正方形)
        pretrained (bool):      是否使用 ImageNet 预训练权重
        alpha (float):          FSA 低频掩码学习范围
        beta (float):           FSA 低频掩码学习范围
        bank_size (int):        全局记忆库大小
        num_anchors (int):      局部记忆库锚点数
        num_samples (int):      对比样本数
        temperature (float):    InfoNCE 温度
        contrastive_weight (float): 对比损失权重 λ
    """

    # ResNet50 各层通道数
    CHANNELS = [256, 512, 1024, 2048]

    def __init__(
        self,
        input_size: int = 256,
        pretrained: bool = True,
        alpha: float = 0.2,
        beta: float = 0.2,
        bank_size: int = 1000,
        num_anchors: int = 32,
        num_samples: int = 16,
        temperature: float = 0.07,
        contrastive_weight: float = 0.1,
    ):
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.input_size = input_size

        # 计算各层特征图尺寸 (ResNet50 对应下采样比例 4,8,16,32)
        strides = [4, 8, 16, 32]
        self.spatial_sizes = [(input_size // s, input_size // s) for s in strides]

        # ---- 编码器 ----
        self.encoder = ResNet50Encoder(pretrained=pretrained)

        # ---- FSA 模块 (各层独立) ----
        self.fsa_modules = nn.ModuleList([
            FrequencyStyleAlignment(
                in_channels=c,
                height=self.spatial_sizes[i][0],
                width=self.spatial_sizes[i][1],
                alpha=alpha, beta=beta
            )
            for i, c in enumerate(self.CHANNELS)
        ])

        # ---- MDFD 解码块 (从深到浅, 4层) ----
        # 最深层: layer4 -> 输出给 layer3
        # 每层的 out_channels 对应下一层的 in_channels
        decoder_out_channels = [256, 512, 1024, 2048]  # 对应 layer1-4 的 C
        self.mdfd_blocks = nn.ModuleList()
        for i in range(len(self.CHANNELS)):
            self.mdfd_blocks.append(
                MDFDBlock(
                    in_channels=self.CHANNELS[i],
                    out_channels=self.CHANNELS[i],
                    num_heads=8
                )
            )

        # ---- 最终预测头 ----
        # 融合最浅层的两个分支输出 -> 变化图
        feat_dim = self.CHANNELS[0]  # 256
        self.prediction_head = nn.Sequential(
            nn.Conv2d(feat_dim * 2, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 1, 1),
        )

        # ---- DMCL ----
        self.dmcl = DualMemoryContrastiveLearning(
            feat_dim=feat_dim,
            bank_size=bank_size,
            num_anchors=num_anchors,
            num_samples=num_samples,
            temperature=temperature
        )

    def forward(self, img_t: torch.Tensor, img_t1: torch.Tensor,
                labels: torch.Tensor = None):
        """
        Args:
            img_t:   时相1图像 [B, 3, H, W]
            img_t1:  时相2图像 [B, 3, H, W]
            labels:  变化标签 [B, H, W] (训练时提供)
        Returns:
            训练模式: (pred_map, total_loss, ce_loss, contrastive_loss)
            推理模式: pred_map [B, 1, H, W] (0~1 概率)
        """
        # ---- 编码器提取双时相特征 ----
        feats_t = self.encoder(img_t)     # 4层特征列表
        feats_t1 = self.encoder(img_t1)

        # ---- 逐层 FSA 风格对齐 ----
        aligned_t, aligned_t1 = [], []
        for i, fsa in enumerate(self.fsa_modules):
            at, at1 = fsa(feats_t[i], feats_t1[i])
            aligned_t.append(at)
            aligned_t1.append(at1)

        # ---- 逐层 MDFD 解码 (从深到浅) ----
        # 使用跳跃连接: 深层输出与浅层FSA对齐特征相加
        cur_t = aligned_t[-1]   # 从最深层开始
        cur_t1 = aligned_t1[-1]

        for i in range(len(self.CHANNELS) - 1, -1, -1):
            O1, O2 = self.mdfd_blocks[i](cur_t, cur_t1)

            if i > 0:
                # 跳跃连接: 与下一层的FSA对齐特征相加
                # 需要调整尺寸
                O1_up = F.interpolate(O1, size=aligned_t[i-1].shape[-2:],
                                      mode='bilinear', align_corners=False)
                O2_up = F.interpolate(O2, size=aligned_t1[i-1].shape[-2:],
                                      mode='bilinear', align_corners=False)

                # 通道对齐 (如果需要)
                if O1_up.shape[1] != aligned_t[i-1].shape[1]:
                    O1_up = O1_up[:, :aligned_t[i-1].shape[1]]
                    O2_up = O2_up[:, :aligned_t1[i-1].shape[1]]

                cur_t = aligned_t[i-1] + O1_up
                cur_t1 = aligned_t1[i-1] + O2_up
            else:
                # 最浅层输出
                final_feat = torch.cat([O1, O2], dim=1)  # [B, 2C, H, W]

        # ---- 预测变化图 ----
        pred_logit = self.prediction_head(final_feat)       # [B, 1, H, W]
        pred_map = torch.sigmoid(pred_logit)                # [B, 1, H, W]

        if self.training and labels is not None:
            # ---- 计算损失 ----
            # 调整标签尺寸
            labels_rs = labels.unsqueeze(1).float()
            if pred_logit.shape[-2:] != labels_rs.shape[-2:]:
                labels_rs = F.interpolate(labels_rs, size=pred_logit.shape[-2:],
                                          mode='nearest')

            # CE/BCE 损失
            ce_loss = F.binary_cross_entropy_with_logits(pred_logit, labels_rs)

            # DMCL 对比损失 (使用最浅层特征)
            feat_for_contrast = O1.detach() if not O1.requires_grad else O1
            labels_for_contrast = labels_rs.squeeze(1).long()
            pred_for_contrast = pred_map.squeeze(1)

            # 特征尺寸与标签对齐
            if feat_for_contrast.shape[-2:] != labels_for_contrast.shape[-2:]:
                feat_for_contrast = F.interpolate(
                    feat_for_contrast, size=labels_for_contrast.shape[-2:],
                    mode='bilinear', align_corners=False
                )
                pred_for_contrast = F.interpolate(
                    pred_for_contrast.unsqueeze(1), size=labels_for_contrast.shape[-2:],
                    mode='bilinear', align_corners=False
                ).squeeze(1)

            contrastive_loss = self.dmcl(feat_for_contrast, labels_for_contrast,
                                          pred_for_contrast)
            total_loss = ce_loss + self.contrastive_weight * contrastive_loss

            return pred_map, total_loss, ce_loss, contrastive_loss

        return pred_map


def build_model(cfg: dict) -> FSAMFD:
    """从配置字典构建模型"""
    return FSAMFD(
        input_size=cfg.get('input_size', 256),
        pretrained=cfg.get('pretrained', True),
        alpha=cfg.get('alpha', 0.2),
        beta=cfg.get('beta', 0.2),
        bank_size=cfg.get('bank_size', 1000),
        num_anchors=cfg.get('num_anchors', 32),
        num_samples=cfg.get('num_samples', 16),
        temperature=cfg.get('temperature', 0.07),
        contrastive_weight=cfg.get('contrastive_weight', 0.1),
    )
