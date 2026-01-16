# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

# ==== 多尺度卷积核与 padding（保持原设定） ====
k1, p1 = 3, 1
k2, p2 = 5, 2
k3, p3 = 9, 4
k4, p4 = 17, 8


# ===================== SE 注意力 =====================
class SELayer1D(nn.Module):
    def __init__(self, nChannels, reduction=16):
        super(SELayer1D, self).__init__()
        self.globalavgpool = nn.AdaptiveAvgPool1d(1)
        mid = max(1, nChannels // reduction)        # 兜底防止为 0
        self.se_block = nn.Sequential(
            nn.Linear(nChannels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, nChannels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):                           # x: [B, C, T]
        alpha = self.globalavgpool(x).squeeze(-1)   # [B, C] 只挤时间维
        alpha = self.se_block(alpha).unsqueeze(-1)  # [B, C, 1]
        return x * alpha


# ===================== 多分支卷积 =====================
class BranchConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BranchConv1D, self).__init__()
        assert out_channels % 4 == 0, \
            f"out_channels({out_channels}) must be divisible by 4 for 4 branches."
        C = out_channels // 4
        self.b1 = nn.Conv1d(in_channels, C, k1, stride, p1, bias=False)
        self.b2 = nn.Conv1d(in_channels, C, k2, stride, p2, bias=False)
        self.b3 = nn.Conv1d(in_channels, C, k3, stride, p3, bias=False)
        self.b4 = nn.Conv1d(in_channels, C, k4, stride, p4, bias=False)

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], dim=1)   # [B, out_channels, T’]


# ===================== 残差基本块 =====================
class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, drop_out_rate, stride):
        super(BasicBlock1D, self).__init__()
        self.operation = nn.Sequential(
            BranchConv1D(in_channels, out_channels, stride),  # 可能下采样
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_out_rate),
            BranchConv1D(out_channels, out_channels, 1),      # 细化
            nn.BatchNorm1d(out_channels),
            SELayer1D(out_channels),
        )

        # 残差分支：对齐长度/通道
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut.add_module('MaxPool', nn.MaxPool1d(stride, ceil_mode=True))
        if in_channels != out_channels:
            self.shortcut.add_module('ShortcutConv', nn.Conv1d(in_channels, out_channels, 1, bias=False))
            self.shortcut.add_module('ShortcutBN', nn.BatchNorm1d(out_channels))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.operation(x)
        sc  = self.shortcut(x)
        return self.relu(out + sc)


# ===================== 主干网络 MSDNN =====================
class MSDNN(nn.Module):
    """
    多尺度卷积 (3/5/9/17) + 残差 + SE + 逐层下采样 + 全局池化
    - 输入:  [B, 54, T]（生理信号多通道序列）
    - 输出:  [B, num_classes]（默认 1；分类时常在外部接 176->K 的线性头）
    """
    def __init__(self,
                 num_classes=1,
                 init_channels=54,
                 growth_rate=16,
                 base_channels=64,
                 stride=2,                 # 每个 block 第一层的 stride（默认都为 2）
                 drop_out_rate=0.2):
        super(MSDNN, self).__init__()

        self.num_channels = init_channels
        block_n = 8
        # 通道进阶：64, 80, 96, 112, 128, 144, 160, 176
        block_c = [base_channels + i * growth_rate for i in range(block_n)]

        self.blocks = nn.Sequential()
        for i, C in enumerate(block_c):
            module = BasicBlock1D(self.num_channels, C, drop_out_rate, stride)
            self.blocks.add_module(f"block{i}", module)
            self.num_channels = C

        # 全局池化 + 线性头（默认 1 维，外部可替换 fc=Identity 后自接分类头）
        self.blocks.add_module("GlobalAvgPool", nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(self.num_channels, num_classes)

    def forward(self, x):                # x: [B, 54, T]
        out = self.blocks(x)             # [B, C_last, 1]
        out = out.squeeze(-1)            # [B, C_last] 只挤掉时间维
        out = self.fc(out)               # [B, num_classes]
        return out
