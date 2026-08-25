"""
CSI Encoder with Dual-Stream Architecture (DualConFi-inspired)

核心思想：
CSI 的"空间结构"（子载波/天线）和"时间结构"是两种物理来源完全不同的信息
- Channel Stream: 建模多径/人体遮挡造成的频域模式
- Temporal Stream: 建模动作节奏、周期性
- Fusion Module: 后期融合产生统一的 CSI 表征

物理解释：
- 子载波维度反映了无线信号在不同频率上的衰减模式（多径效应）
- 时间维度反映了人体动作的动态变化
- 分离建模可以让网络更好地学习这两种不同的物理特性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelStream(nn.Module):
    """
    Channel Stream: 对每个时间步的子载波维度进行建模
    
    输入: [B, T, Subcarriers, Antennas]
    输出: [B, T, channel_feat_dim]
    
    物理意义：
    - 捕捉不同子载波、天线间的相关性
    - 学习多径效应和人体遮挡造成的频域模式
    """
    
    def __init__(self, num_subcarriers, num_antennas, channel_dim):
        super().__init__()
        
        self.num_subcarriers = num_subcarriers
        self.num_antennas = num_antennas
        
        # 将 (Subcarriers, Antennas) 展平为特征向量
        input_dim = num_subcarriers * num_antennas
        
        # 多层感知机提取频域特征
        self.channel_net = nn.Sequential(
            nn.Linear(input_dim, channel_dim * 2),
            nn.BatchNorm1d(channel_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(channel_dim * 2, channel_dim),
            nn.BatchNorm1d(channel_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, T, Subcarriers, Antennas]
        Returns:
            channel_features: [B, T, channel_dim]
        """
        B, T, S, A = x.shape
        
        # 展平子载波和天线维度
        x_flat = x.reshape(B * T, S * A)  # [B*T, S*A]
        
        # 提取频域特征
        channel_feat = self.channel_net(x_flat)  # [B*T, channel_dim]
        
        # 恢复时间维度
        channel_feat = channel_feat.reshape(B, T, -1)  # [B, T, channel_dim]
        
        return channel_feat


class TemporalStream(nn.Module):
    """
    Temporal Stream: 对时间序列进行建模
    
    输入: [B, T, feature_dim]
    输出: [B, temporal_dim]
    
    物理意义：
    - 捕捉动作的时序动态
    - 学习动作的节奏、周期性、持续时间等时间特性
    """
    
    def __init__(self, input_dim, temporal_dim, num_layers=2):
        super().__init__()
        
        # 使用 Temporal Convolutional Network (TCN)
        self.tcn = nn.ModuleList()
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else temporal_dim
            self.tcn.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, temporal_dim, kernel_size=3, padding=1),
                    nn.BatchNorm1d(temporal_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1)
                )
            )
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, feature_dim]
        Returns:
            temporal_features: [B, temporal_dim]
        """
        # 转换为 Conv1d 格式: [B, feature_dim, T]
        x = x.transpose(1, 2)  # [B, feature_dim, T]
        
        # TCN 层
        for tcn_layer in self.tcn:
            x = tcn_layer(x)  # [B, temporal_dim, T]
        
        # 全局池化
        x = self.global_pool(x)  # [B, temporal_dim, 1]
        x = x.squeeze(-1)  # [B, temporal_dim]
        
        return x


class FusionModule(nn.Module):
    """
    Fusion Module: 融合 Channel 和 Temporal 特征
    
    输入: channel_features [B, T, channel_dim], temporal_features [B, temporal_dim]
    输出: fused_features [B, fusion_dim]
    
    策略：
    - Late fusion: 先分别提取特征，再融合
    - Concat + MLP 进行特征融合
    """
    
    def __init__(self, channel_dim, temporal_dim, fusion_dim):
        super().__init__()
        
        # 对 channel features 进行时间池化
        self.channel_pool = nn.AdaptiveAvgPool1d(1)
        
        # 融合网络
        concat_dim = channel_dim + temporal_dim
        self.fusion_net = nn.Sequential(
            nn.Linear(concat_dim, fusion_dim * 2),
            nn.BatchNorm1d(fusion_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, channel_feat, temporal_feat):
        """
        Args:
            channel_feat: [B, T, channel_dim]
            temporal_feat: [B, temporal_dim]
        Returns:
            fused_feat: [B, fusion_dim]
        """
        # 对 channel features 进行时间池化
        channel_feat = channel_feat.transpose(1, 2)  # [B, channel_dim, T]
        channel_feat = self.channel_pool(channel_feat)  # [B, channel_dim, 1]
        channel_feat = channel_feat.squeeze(-1)  # [B, channel_dim]
        
        # 拼接
        concat_feat = torch.cat([channel_feat, temporal_feat], dim=1)  # [B, channel_dim + temporal_dim]
        
        # 融合
        fused_feat = self.fusion_net(concat_feat)  # [B, fusion_dim]
        
        return fused_feat


class CSIEncoder(nn.Module):
    """
    CSI Dual-Stream Encoder
    
    架构：
    1. Channel Stream: 提取频域特征（子载波/天线相关性）
    2. Temporal Stream: 提取时序特征（动作动态）
    3. Fusion Module: 后期融合
    
    输入: [B, T, Subcarriers, Antennas]
    输出: [B, fusion_dim]
    """
    
    def __init__(self, 
                 num_subcarriers=30,
                 num_antennas=3,
                 channel_dim=128,
                 temporal_dim=128,
                 fusion_dim=256):
        """
        Args:
            num_subcarriers: 子载波数量
            num_antennas: 天线数量
            channel_dim: Channel Stream 输出维度
            temporal_dim: Temporal Stream 输出维度
            fusion_dim: 融合后的特征维度
        """
        super().__init__()
        
        self.num_subcarriers = num_subcarriers
        self.num_antennas = num_antennas
        
        # Channel Stream: 建模频域结构
        self.channel_stream = ChannelStream(num_subcarriers, num_antennas, channel_dim)
        
        # Temporal Stream: 建模时序结构
        # 输入是 channel stream 的输出
        self.temporal_stream = TemporalStream(channel_dim, temporal_dim, num_layers=2)
        
        # Fusion Module: 融合两个流
        self.fusion_module = FusionModule(channel_dim, temporal_dim, fusion_dim)
        
        self.fusion_dim = fusion_dim
    
    def forward(self, x):
        """
        Args:
            x: [B, T, Subcarriers, Antennas] 或 [B, T, C]
        Returns:
            fused_feature: [B, fusion_dim]
        """
        # 如果输入是 [B, T, C]，需要 reshape
        if x.dim() == 3:
            B, T, C = x.shape
            # 假设 C = Subcarriers * Antennas
            x = x.reshape(B, T, self.num_subcarriers, self.num_antennas)
        
        # 1. Channel Stream: 提取频域特征
        channel_feat = self.channel_stream(x)  # [B, T, channel_dim]
        
        # 2. Temporal Stream: 提取时序特征
        temporal_feat = self.temporal_stream(channel_feat)  # [B, temporal_dim]
        
        # 3. Fusion: 融合两个流
        fused_feat = self.fusion_module(channel_feat, temporal_feat)  # [B, fusion_dim]
        
        return fused_feat
    
    def get_output_dim(self):
        """返回输出特征维度"""
        return self.fusion_dim


# 测试代码
if __name__ == '__main__':
    # 测试 CSI Encoder
    batch_size = 4
    T = 100  # 时间步
    num_subcarriers = 30
    num_antennas = 3
    
    # 创建模型
    model = CSIEncoder(
        num_subcarriers=num_subcarriers,
        num_antennas=num_antennas,
        channel_dim=128,
        temporal_dim=128,
        fusion_dim=256
    )
    
    # 创建输入
    x = torch.randn(batch_size, T, num_subcarriers, num_antennas)
    
    # 前向传播
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
