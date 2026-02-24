"""
Multi-Dimensional Feature Decoding (MDFD) Module
论文: Frequency-domain style alignment for change detection via multi-dimensional feature decoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# 分层多尺度自注意力 (用于差异特征和融合特征)
# ---------------------------------------------------------------------------

class HierarchicalMultiScaleSelfAttention(nn.Module):
    """
    分层多尺度自注意力机制
    
    差异特征和融合特征采用Query互换的交叉注意力策略:
    - 融合特征的 Query 去查询差异特征的 Key-Value
    - 差异特征的 Query 去查询融合特征的 Key-Value
    """

    def __init__(self, diff_channels: int, fuse_channels: int,
                 num_heads: int = 8, num_levels: int = 3):
        super().__init__()
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.diff_channels = diff_channels
        self.fuse_channels = fuse_channels

        dh_diff = diff_channels // num_heads
        dh_fuse = fuse_channels // num_heads
        self.scale_diff = math.sqrt(dh_diff)
        self.scale_fuse = math.sqrt(dh_fuse)

        # 差异特征各层 QKV 线性层
        self.diff_qkv = nn.ModuleList([
            nn.Linear(diff_channels, diff_channels * 3) for _ in range(num_levels)
        ])
        # 融合特征各层 QKV 线性层
        self.fuse_qkv = nn.ModuleList([
            nn.Linear(fuse_channels, fuse_channels * 3) for _ in range(num_levels)
        ])

        # 聚合卷积
        self.diff_aggregate = nn.Conv2d(diff_channels * num_levels, diff_channels, 3, padding=1)
        self.fuse_aggregate = nn.Conv2d(fuse_channels * num_levels, fuse_channels, 3, padding=1)

        self.norm_diff = nn.LayerNorm(diff_channels)
        self.norm_fuse = nn.LayerNorm(fuse_channels)

    def _multi_scale_decompose(self, feat: torch.Tensor, level: int):
        """
        多尺度分解: level=1 原始分辨率, level=2,3 下采样后上采样
        """
        B, C, H, W = feat.shape
        if level == 1:
            return feat
        else:
            pool_size = H // (2 ** level)
            if pool_size < 1:
                pool_size = 1
            down = F.adaptive_avg_pool2d(feat, pool_size)
            up = F.interpolate(down, size=(H, W), mode='bilinear', align_corners=False)
            return up

    def _multi_head_attention(self, Q, K, V, num_heads, scale):
        """标准多头注意力"""
        B, N, C = Q.shape
        head_dim = C // num_heads

        Q = Q.view(B, N, num_heads, head_dim).transpose(1, 2)  # [B, h, N, d]
        K = K.view(B, N, num_heads, head_dim).transpose(1, 2)
        V = V.view(B, N, num_heads, head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)   # [B, h, N, d]
        out = out.transpose(1, 2).reshape(B, N, C)
        return out

    def forward(self, F_diff: torch.Tensor, F_fuse: torch.Tensor):
        """
        Args:
            F_diff: 差异特征 [B, C, H, W]
            F_fuse: 融合特征 [B, 2C, H, W]
        Returns:
            F_diff_star, F_fuse_star: 增强后的特征
        """
        B, C, H, W = F_diff.shape
        diff_outs, fuse_outs = [], []

        for l in range(self.num_levels):
            level = l + 1

            # 多尺度分解
            fd_l = self._multi_scale_decompose(F_diff, level)  # [B, C, H, W]
            ff_l = self._multi_scale_decompose(F_fuse, level)  # [B, 2C, H, W]

            # Flatten 为序列
            fd_flat = fd_l.flatten(2).transpose(1, 2)  # [B, N, C]
            ff_flat = ff_l.flatten(2).transpose(1, 2)  # [B, N, 2C]

            # LayerNorm
            fd_flat = self.norm_diff(fd_flat)
            # 注: fuse 使用不同 norm 需要对应通道数
            # 简化: 直接用 Linear 投影

            # 生成 QKV
            fd_qkv = self.diff_qkv[l](fd_flat)
            Qd, Kd, Vd = fd_qkv.chunk(3, dim=-1)

            ff_qkv = self.fuse_qkv[l](ff_flat)
            Qf, Kf, Vf = ff_qkv.chunk(3, dim=-1)

            # Query 互换交叉注意力
            # 差异特征增强: 融合特征 Query 查询差异特征 K-V
            attn_d = self._multi_head_attention(Qf, Kd, Vd,
                                                 self.num_heads, self.scale_diff)
            # 融合特征增强: 差异特征 Query 查询融合特征 K-V
            attn_f = self._multi_head_attention(Qd, Kf, Vf,
                                                 self.num_heads, self.scale_fuse)

            # Reshape 回 [B, C, H, W]
            attn_d = attn_d.transpose(1, 2).view(B, C, H, W)
            attn_f = attn_f.transpose(1, 2).view(B, self.fuse_channels, H, W)

            diff_outs.append(attn_d)
            fuse_outs.append(attn_f)

        # 多层聚合
        F_diff_star = self.diff_aggregate(torch.cat(diff_outs, dim=1))
        F_fuse_star = self.fuse_aggregate(torch.cat(fuse_outs, dim=1))

        return F_diff_star, F_fuse_star


# ---------------------------------------------------------------------------
# 变化感知多头交叉注意力 (用于时序特征)
# ---------------------------------------------------------------------------

class ChangeAwareCrossAttention(nn.Module):
    """
    变化感知多头交叉注意力机制
    
    使用差异特征和融合特征生成变化感知掩码，
    引导双时相原始特征的双向交叉注意力。
    """

    def __init__(self, in_channels: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        head_dim = in_channels // num_heads
        self.scale = math.sqrt(head_dim)

        # 变化感知掩码生成
        self.mask_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 1),  # diff + fuse(2C) -> C
            nn.Sigmoid()
        )

        # 时序特征 QKV 映射
        self.qkv_t = nn.Linear(in_channels, in_channels * 3)
        self.qkv_t1 = nn.Linear(in_channels, in_channels * 3)

        # 输出聚合
        self.out_conv = nn.Conv2d(in_channels * 2, in_channels * 2, 3, padding=1)

    def forward(self, feat_t: torch.Tensor, feat_t1: torch.Tensor,
                F_diff: torch.Tensor, F_fuse: torch.Tensor):
        """
        Args:
            feat_t:  时相1特征 [B, C, H, W]
            feat_t1: 时相2特征 [B, C, H, W]
            F_diff:  差异特征  [B, C, H, W]
            F_fuse:  融合特征  [B, 2C, H, W]
        Returns:
            F_cross: 交叉注意力特征 [B, 2C, H, W]
        """
        B, C, H, W = feat_t.shape
        h = self.num_heads
        head_dim = C // h

        # 生成变化感知掩码
        M_c = self.mask_conv(torch.cat([F_diff, F_fuse], dim=1))  # [B, C, H, W]
        M_c_flat = M_c.flatten(2).transpose(1, 2)                  # [B, N, C]

        # Flatten 时序特征
        Et_flat = feat_t.flatten(2).transpose(1, 2)    # [B, N, C]
        Et1_flat = feat_t1.flatten(2).transpose(1, 2)

        # QKV 映射
        Qt, Kt, Vt = self.qkv_t(Et_flat).chunk(3, dim=-1)
        Qt1, Kt1, Vt1 = self.qkv_t1(Et1_flat).chunk(3, dim=-1)

        # 变化感知掩码应用到 Value
        Vt_masked = Vt * M_c_flat
        Vt1_masked = Vt1 * M_c_flat

        # 多头注意力计算
        def mha(Q, K, V):
            N = Q.shape[1]
            Q = Q.view(B, N, h, head_dim).transpose(1, 2)
            K = K.view(B, N, h, head_dim).transpose(1, 2)
            V = V.view(B, N, h, head_dim).transpose(1, 2)
            attn = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / self.scale, dim=-1)
            return torch.matmul(attn, V).transpose(1, 2).reshape(B, N, C)

        # 双向交叉注意力
        Attn_t_from_t1 = mha(Qt, Kt1, Vt1_masked)   # t1引导t
        Attn_t1_from_t = mha(Qt1, Kt, Vt_masked)    # t引导t1

        # Reshape 回 spatial
        A_t = Attn_t_from_t1.transpose(1, 2).view(B, C, H, W)
        A_t1 = Attn_t1_from_t.transpose(1, 2).view(B, C, H, W)

        # 融合双向结果
        F_cross = self.out_conv(torch.cat([A_t, A_t1], dim=1))  # [B, 2C, H, W]
        return F_cross


# ---------------------------------------------------------------------------
# MDFD 完整解码块
# ---------------------------------------------------------------------------

class MDFDBlock(nn.Module):
    """
    多维特征解码块 (单层)
    
    构建四维特征张量空间 (原始特征、差异特征、融合特征、频域变换特征)，
    并采用双路协作注意力机制。
    """

    def __init__(self, in_channels: int, out_channels: int,
                 num_heads: int = 8):
        super().__init__()
        self.in_channels = in_channels
        fuse_channels = in_channels * 2

        # 融合特征降维 (Concat后2C -> C)
        self.fuse_proj = nn.Conv2d(fuse_channels, in_channels, 1)

        # 双路注意力
        self.hmsa = HierarchicalMultiScaleSelfAttention(
            diff_channels=in_channels,
            fuse_channels=in_channels,  # 降维后
            num_heads=num_heads
        )
        self.ca = ChangeAwareCrossAttention(in_channels, num_heads)

        # 特征融合: diff*(C) + fuse*(C) + cross(2C) -> 4C
        total_c = in_channels + in_channels + fuse_channels
        # 转置卷积输出两个分支
        self.branch1 = nn.ConvTranspose2d(total_c, out_channels, 3, padding=1)
        self.branch2 = nn.ConvTranspose2d(total_c, out_channels, 3, padding=1)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat_t: torch.Tensor, feat_t1: torch.Tensor):
        """
        Args:
            feat_t:  时相1对齐特征 [B, C, H, W]
            feat_t1: 时相2对齐特征 [B, C, H, W]
        Returns:
            O1, O2: 两个输出分支 [B, out_C, H, W]
        """
        # 构建多维特征空间
        F_diff = torch.abs(feat_t - feat_t1)                       # 差异特征 [B, C, H, W]
        F_fuse_raw = torch.cat([feat_t, feat_t1], dim=1)           # 融合特征 [B, 2C, H, W]
        F_fuse = self.fuse_proj(F_fuse_raw)                        # 降维 [B, C, H, W]

        # 双路协作注意力
        F_diff_star, F_fuse_star = self.hmsa(F_diff, F_fuse)       # 多尺度自注意力路径
        F_cross = self.ca(feat_t, feat_t1, F_diff, F_fuse_raw)     # 交叉注意力路径

        # 特征聚合
        combined = torch.cat([F_diff_star, F_fuse_star, F_cross], dim=1)

        # 两个输出分支 (供跳跃连接)
        O1 = self.relu(self.bn(self.branch1(combined)))
        O2 = self.relu(self.bn(self.branch2(combined)))

        return O1, O2
