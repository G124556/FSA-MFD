"""
Frequency-domain Style Alignment (FSA) Module
论文: Frequency-domain style alignment for change detection via multi-dimensional feature decoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyStyleAlignment(nn.Module):
    """
    频域风格对齐模块 (FSA)
    
    通过可学习的高斯分布掩码自适应地学习最优幅度谱交换策略，
    实现双时相特征的双向风格一致性对齐。
    
    Args:
        in_channels (int): 输入特征通道数
        height (int): 特征图高度
        width (int): 特征图宽度
        alpha (float): 低频区域学习范围超参数 (默认 0.2)
        beta (float): 低频区域学习范围超参数 (默认 0.2)
    """

    def __init__(self, in_channels: int, height: int, width: int,
                 alpha: float = 0.2, beta: float = 0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.H = height
        self.W = width

        # 为双时相各设计独立的可学习频域掩码参数
        self.theta_t = nn.Parameter(torch.zeros(height, width))
        self.theta_t1 = nn.Parameter(torch.zeros(height, width))

        # 限制掩码学习区域为低频区域
        self._low_freq_mask = self._build_low_freq_region(height, width, alpha, beta)

    def _build_low_freq_region(self, H: int, W: int, alpha: float, beta: float) -> torch.Tensor:
        """构建低频区域布尔掩码 (中心化)"""
        mask = torch.zeros(H, W, dtype=torch.bool)
        h_range = int(alpha * H)
        w_range = int(beta * W)
        cx, cy = H // 2, W // 2
        mask[cx - h_range: cx + h_range, cy - w_range: cy + w_range] = True
        return mask

    def forward(self, feat_t: torch.Tensor, feat_t1: torch.Tensor):
        """
        Args:
            feat_t:  时相1特征 [B, C, H, W]
            feat_t1: 时相2特征 [B, C, H, W]
        Returns:
            aligned_t, aligned_t1: 风格对齐后的双时相特征
        """
        device = feat_t.device

        # 确保低频掩码在正确的设备上
        low_freq_mask = self._low_freq_mask.to(device)

        # --- 2D FFT 频谱分解 ---
        F_t = torch.fft.fft2(feat_t, norm='ortho')     # [B, C, H, W] complex
        F_t1 = torch.fft.fft2(feat_t1, norm='ortho')

        # 将频谱中心化 (低频移至中心)
        F_t = torch.fft.fftshift(F_t, dim=(-2, -1))
        F_t1 = torch.fft.fftshift(F_t1, dim=(-2, -1))

        # 分离幅度谱和相位谱
        A_t = torch.abs(F_t)    # 幅度谱 (编码风格信息)
        P_t = torch.angle(F_t)  # 相位谱 (编码语义结构)
        A_t1 = torch.abs(F_t1)
        P_t1 = torch.angle(F_t1)

        # --- 可学习掩码 (仅在低频区域学习) ---
        # 只激活低频区域的梯度，高频区域设为0
        theta_t_masked = self.theta_t * low_freq_mask.float()
        theta_t1_masked = self.theta_t1 * low_freq_mask.float()

        M_t = torch.sigmoid(theta_t_masked)    # [H, W]
        M_t1 = torch.sigmoid(theta_t1_masked)  # [H, W]

        # 扩展到 [B, C, H, W] 以进行广播
        M_t = M_t.unsqueeze(0).unsqueeze(0)    # [1, 1, H, W]
        M_t1 = M_t1.unsqueeze(0).unsqueeze(0)

        # --- 双向幅度谱互换 ---
        # 时相1: 部分引入时相2的幅度谱 (风格迁移)
        A_aligned_t = M_t * A_t1 + (1 - M_t) * A_t
        # 时相2: 部分引入时相1的幅度谱 (风格迁移)
        A_aligned_t1 = M_t1 * A_t + (1 - M_t1) * A_t1

        # --- 重建频域表示并做逆FFT ---
        F_aligned_t = A_aligned_t * torch.exp(1j * P_t)
        F_aligned_t1 = A_aligned_t1 * torch.exp(1j * P_t1)

        # 逆中心化
        F_aligned_t = torch.fft.ifftshift(F_aligned_t, dim=(-2, -1))
        F_aligned_t1 = torch.fft.ifftshift(F_aligned_t1, dim=(-2, -1))

        # 逆FFT 返回空间域
        feat_aligned_t = torch.fft.ifft2(F_aligned_t, norm='ortho').real
        feat_aligned_t1 = torch.fft.ifft2(F_aligned_t1, norm='ortho').real

        return feat_aligned_t, feat_aligned_t1


class FSAWrapper(nn.Module):
    """
    对编码器各层分别应用FSA的包装器
    自动根据特征图尺寸创建FSA实例
    """

    def __init__(self, channels_list: list, spatial_sizes: list,
                 alpha: float = 0.2, beta: float = 0.2):
        """
        Args:
            channels_list: 各层通道数列表, e.g., [64, 256, 512, 1024, 2048]
            spatial_sizes: 各层 (H, W) 列表
            alpha, beta: 低频掩码超参数
        """
        super().__init__()
        self.fsa_modules = nn.ModuleList([
            FrequencyStyleAlignment(c, h, w, alpha, beta)
            for c, (h, w) in zip(channels_list, spatial_sizes)
        ])

    def forward(self, feats_t: list, feats_t1: list):
        """
        Args:
            feats_t:  时相1各层特征列表
            feats_t1: 时相2各层特征列表
        Returns:
            aligned_t_list, aligned_t1_list
        """
        aligned_t, aligned_t1 = [], []
        for i, fsa in enumerate(self.fsa_modules):
            at, at1 = fsa(feats_t[i], feats_t1[i])
            aligned_t.append(at)
            aligned_t1.append(at1)
        return aligned_t, aligned_t1
