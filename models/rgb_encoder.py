"""
RGB Skeleton Encoder (Momentum Key Encoder)

核心角色：
- 作为 MoCo-style 对比学习中的 Key Encoder
- 提供稳定的语义锚点，引导 CSI 学习鲁棒表示
- 参数通过 EMA 从 CSI Encoder 更新，不参与梯度反向传播

架构设计：
- 与 CSI Encoder 结构语义对齐（便于 EMA 更新）
- 输入：RGB 骨架序列 [B, T, J, 3]
- 输出：语义 embedding [B, fusion_dim]

关键约束（必须严格遵守）：
1. 所有参数 requires_grad = False
2. 不加入 optimizer
3. 仅通过 EMA 更新参数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkeletonSpatialStream(nn.Module):
    """
    Skeleton Spatial Stream: 对每个时间步的骨架空间结构进行建模
    
    输入: [B, T, J, 3]  (J=关节数, 3=xyz坐标)
    输出: [B, T, spatial_dim]
    
    物理意义：
    - 捕捉关节间的空间关系
    - 学习人体姿态的空间结构
    """
    
    def __init__(self, num_joints, coord_dim, spatial_dim):
        super().__init__()
        
        self.num_joints = num_joints
        self.coord_dim = coord_dim
        
        # 将 (J, 3) 展平为特征向量
        input_dim = num_joints * coord_dim
        
        # 空间特征提取网络
        self.spatial_net = nn.Sequential(
            nn.Linear(input_dim, spatial_dim * 2),
            nn.BatchNorm1d(spatial_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(spatial_dim * 2, spatial_dim),
            nn.BatchNorm1d(spatial_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, T, J, 3]
        Returns:
            spatial_features: [B, T, spatial_dim]
        """
        B, T, J, C = x.shape
        
        # 展平关节和坐标维度
        x_flat = x.reshape(B * T, J * C)  # [B*T, J*3]
        
        # 提取空间特征
        spatial_feat = self.spatial_net(x_flat)  # [B*T, spatial_dim]
        
        # 恢复时间维度
        spatial_feat = spatial_feat.reshape(B, T, -1)  # [B, T, spatial_dim]
        
        return spatial_feat


class SkeletonTemporalStream(nn.Module):
    """
    Skeleton Temporal Stream: 对骨架时间序列进行建模
    
    输入: [B, T, feature_dim]
    输出: [B, temporal_dim]
    
    物理意义：
    - 捕捉动作的时序动态
    - 学习动作的节奏、周期性
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


class SkeletonFusionModule(nn.Module):
    """
    Skeleton Fusion Module: 融合空间和时间特征
    
    输入: spatial_features [B, T, spatial_dim], temporal_features [B, temporal_dim]
    输出: fused_features [B, fusion_dim]
    """
    
    def __init__(self, spatial_dim, temporal_dim, fusion_dim):
        super().__init__()
        
        # 对 spatial features 进行时间池化
        self.spatial_pool = nn.AdaptiveAvgPool1d(1)
        
        # 融合网络
        concat_dim = spatial_dim + temporal_dim
        self.fusion_net = nn.Sequential(
            nn.Linear(concat_dim, fusion_dim * 2),
            nn.BatchNorm1d(fusion_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, spatial_feat, temporal_feat):
        """
        Args:
            spatial_feat: [B, T, spatial_dim]
            temporal_feat: [B, temporal_dim]
        Returns:
            fused_feat: [B, fusion_dim]
        """
        # 对 spatial features 进行时间池化
        spatial_feat = spatial_feat.transpose(1, 2)  # [B, spatial_dim, T]
        spatial_feat = self.spatial_pool(spatial_feat)  # [B, spatial_dim, 1]
        spatial_feat = spatial_feat.squeeze(-1)  # [B, spatial_dim]
        
        # 拼接
        concat_feat = torch.cat([spatial_feat, temporal_feat], dim=1)
        
        # 融合
        fused_feat = self.fusion_net(concat_feat)  # [B, fusion_dim]
        
        return fused_feat


class RGBEncoder(nn.Module):
    """
    RGB Skeleton Encoder (Momentum Key Encoder)
    
    架构（与 CSI Encoder 语义对齐）：
    1. Spatial Stream: 提取空间特征（关节空间关系）
    2. Temporal Stream: 提取时序特征（动作动态）
    3. Fusion Module: 后期融合
    
    输入: [B, T, J, 3]
    输出: [B, fusion_dim]
    
    关键约束：
    - 所有参数 requires_grad = False
    - 不加入 optimizer
    - 仅通过 EMA 更新参数
    """
    
    def __init__(self,
                 num_joints=17,
                 coord_dim=3,
                 spatial_dim=128,
                 temporal_dim=128,
                 fusion_dim=256):
        """
        Args:
            num_joints: 关节数量
            coord_dim: 坐标维度（通常为 3）
            spatial_dim: Spatial Stream 输出维度
            temporal_dim: Temporal Stream 输出维度
            fusion_dim: 融合后的特征维度
        """
        super().__init__()
        
        self.num_joints = num_joints
        self.coord_dim = coord_dim
        
        # Spatial Stream: 建模空间结构
        self.spatial_stream = SkeletonSpatialStream(num_joints, coord_dim, spatial_dim)
        
        # Temporal Stream: 建模时序结构
        self.temporal_stream = SkeletonTemporalStream(spatial_dim, temporal_dim, num_layers=2)
        
        # Fusion Module: 融合两个流
        self.fusion_module = SkeletonFusionModule(spatial_dim, temporal_dim, fusion_dim)
        
        self.fusion_dim = fusion_dim
        
        # 【关键】冻结所有参数
        self._freeze_parameters()
    
    def _freeze_parameters(self):
        """
        冻结所有参数
        
        【禁止项】RGB Encoder 的参数不参与梯度反向传播
        """
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        Args:
            x: [B, T, J, 3]
        Returns:
            fused_feature: [B, fusion_dim]
        
        【注意】调用时必须使用 torch.no_grad() 包裹
        """
        # 1. Spatial Stream: 提取空间特征
        spatial_feat = self.spatial_stream(x)  # [B, T, spatial_dim]
        
        # 2. Temporal Stream: 提取时序特征
        temporal_feat = self.temporal_stream(spatial_feat)  # [B, temporal_dim]
        
        # 3. Fusion: 融合两个流
        fused_feat = self.fusion_module(spatial_feat, temporal_feat)  # [B, fusion_dim]
        
        return fused_feat
    
    def get_output_dim(self):
        """返回输出特征维度"""
        return self.fusion_dim


def create_rgb_encoder_from_csi_encoder(csi_encoder, num_joints=17, coord_dim=3):
    """
    从 CSI Encoder 创建结构对齐的 RGB Encoder
    
    Args:
        csi_encoder: CSI Encoder 实例
        num_joints: 关节数量
        coord_dim: 坐标维度
    
    Returns:
        rgb_encoder: RGB Encoder 实例（参数已冻结）
    """
    # 获取 CSI Encoder 的维度配置
    # 注意：这里假设 CSI Encoder 有 channel_dim, temporal_dim, fusion_dim 属性
    # 实际实现中需要根据 CSI Encoder 的具体结构调整
    
    rgb_encoder = RGBEncoder(
        num_joints=num_joints,
        coord_dim=coord_dim,
        spatial_dim=128,  # 对应 CSI 的 channel_dim
        temporal_dim=128,  # 对应 CSI 的 temporal_dim
        fusion_dim=256     # 对应 CSI 的 fusion_dim
    )
    
    # 确保参数已冻结
    assert all(not p.requires_grad for p in rgb_encoder.parameters()), \
        "RGB Encoder 参数必须被冻结"
    
    return rgb_encoder


# 测试代码
if __name__ == '__main__':
    # 测试 RGB Encoder
    batch_size = 4
    T = 100  # 时间步
    num_joints = 17
    coord_dim = 3
    
    # 创建模型
    model = RGBEncoder(
        num_joints=num_joints,
        coord_dim=coord_dim,
        spatial_dim=128,
        temporal_dim=128,
        fusion_dim=256
    )
    
    # 验证参数已冻结
    assert all(not p.requires_grad for p in model.parameters()), \
        "RGB Encoder 参数必须被冻结"
    
    # 创建输入
    x = torch.randn(batch_size, T, num_joints, coord_dim)
    
    # 前向传播（必须使用 no_grad）
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"All parameters frozen: {all(not p.requires_grad for p in model.parameters())}")
