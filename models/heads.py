"""
Projection, Classification, and Regression Heads

模块说明：
1. Projector: 将编码器输出映射到对比学习空间，输出 L2 归一化
2. ClassifierHead: 活动分类头，输出 logits
3. RegressorHead: 骨架回归头（辅助任务，仅训练时使用）

关键约束：
- Projector 输出必须 L2 归一化
- 推理阶段不使用 Projector 和 RegressorHead
- 只有 ClassifierHead 在推理时使用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    """
    投影头：将编码器特征映射到对比学习空间
    
    架构: Linear -> BN -> ReLU -> Linear -> L2 Normalize
    
    输入: [B, fusion_dim]
    输出: [B, projector_dim]（L2 归一化）
    
    用途：
    - 对比学习中的特征投影
    - 训练时使用，推理时移除
    """
    
    def __init__(self, input_dim, hidden_dim=None, output_dim=128):
        """
        Args:
            input_dim: 输入特征维度（编码器输出维度）
            hidden_dim: 隐藏层维度，默认为 input_dim
            output_dim: 输出维度（投影空间维度）
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.output_dim = output_dim
    
    def forward(self, x):
        """
        Args:
            x: [B, input_dim]
        Returns:
            z: [B, output_dim]（L2 归一化）
        """
        z = self.projector(x)
        
        # L2 归一化（对比学习的关键步骤）
        z = F.normalize(z, dim=1, p=2)
        
        return z


class ClassifierHead(nn.Module):
    """
    分类头：活动分类任务
    
    架构: Linear -> BN -> ReLU -> Dropout -> Linear
    
    输入: [B, fusion_dim]
    输出: [B, num_classes]（logits）
    
    用途：
    - 主任务：活动分类
    - 训练和推理时都使用
    """
    
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.1):
        """
        Args:
            input_dim: 输入特征维度（编码器输出维度）
            hidden_dim: 隐藏层维度
            num_classes: 类别数量
            dropout: Dropout 比例
        """
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x):
        """
        Args:
            x: [B, input_dim]
        Returns:
            logits: [B, num_classes]
        """
        logits = self.classifier(x)
        return logits


class RegressorHead(nn.Module):
    """
    骨架回归头：辅助任务，预测 3D 骨架关键点
    
    架构: Linear -> BN -> ReLU -> Dropout -> Linear
    
    输入: [B, fusion_dim]
    输出: [B, num_joints * 3]（3D 关节坐标）
    
    用途：
    - 辅助任务：骨架回归
    - 仅训练时使用，推理时移除
    - 帮助 CSI 编码器学习与骨架相关的特征
    """
    
    def __init__(self, input_dim, hidden_dim, num_joints, coord_dim=3, dropout=0.1):
        """
        Args:
            input_dim: 输入特征维度（编码器输出维度）
            hidden_dim: 隐藏层维度
            num_joints: 关节数量
            coord_dim: 坐标维度（通常为 3）
            dropout: Dropout 比例
        """
        super().__init__()
        
        self.num_joints = num_joints
        self.coord_dim = coord_dim
        output_dim = num_joints * coord_dim
        
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, input_dim]
        Returns:
            joints: [B, num_joints * 3] 或 [B, num_joints, 3]
        """
        joints = self.regressor(x)
        
        # 可选：reshape 为 [B, num_joints, 3]
        # joints = joints.reshape(-1, self.num_joints, self.coord_dim)
        
        return joints
    
    def forward_reshaped(self, x):
        """
        返回 reshape 后的关节坐标
        
        Args:
            x: [B, input_dim]
        Returns:
            joints: [B, num_joints, 3]
        """
        joints = self.regressor(x)
        joints = joints.reshape(-1, self.num_joints, self.coord_dim)
        return joints


class HeadsModule(nn.Module):
    """
    头部模块集合：包含所有任务头
    
    包含：
    - CSI Projector: CSI 特征投影
    - RGB Projector: RGB 特征投影
    - Classifier: 活动分类
    - Regressor: 骨架回归（辅助任务）
    
    关键约束：
    - RGB Projector 参数需要冻结（与 RGB Encoder 一致）
    - 推理时只使用 Classifier
    """
    
    def __init__(self,
                 fusion_dim,
                 projector_dim=128,
                 classifier_hidden=256,
                 regressor_hidden=256,
                 num_classes=27,
                 num_joints=17,
                 dropout=0.1):
        """
        Args:
            fusion_dim: 编码器输出维度
            projector_dim: 投影空间维度
            classifier_hidden: 分类器隐藏层维度
            regressor_hidden: 回归器隐藏层维度
            num_classes: 类别数量
            num_joints: 关节数量
            dropout: Dropout 比例
        """
        super().__init__()
        
        # CSI Projector（参与梯度更新）
        self.csi_projector = Projector(
            input_dim=fusion_dim,
            hidden_dim=fusion_dim,
            output_dim=projector_dim
        )
        
        # RGB Projector（参数冻结，通过 EMA 更新）
        self.rgb_projector = Projector(
            input_dim=fusion_dim,
            hidden_dim=fusion_dim,
            output_dim=projector_dim
        )
        # 冻结 RGB Projector 参数
        self._freeze_rgb_projector()
        
        # Classifier（活动分类）
        self.classifier = ClassifierHead(
            input_dim=fusion_dim,
            hidden_dim=classifier_hidden,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # Regressor（骨架回归，辅助任务）
        self.regressor = RegressorHead(
            input_dim=fusion_dim,
            hidden_dim=regressor_hidden,
            num_joints=num_joints,
            dropout=dropout
        )
        
        self.fusion_dim = fusion_dim
        self.projector_dim = projector_dim
        self.num_classes = num_classes
        self.num_joints = num_joints
    
    def _freeze_rgb_projector(self):
        """冻结 RGB Projector 参数"""
        for param in self.rgb_projector.parameters():
            param.requires_grad = False
    
    def project_csi(self, csi_feat):
        """
        CSI 特征投影
        
        Args:
            csi_feat: [B, fusion_dim]
        Returns:
            z_csi: [B, projector_dim]（L2 归一化）
        """
        return self.csi_projector(csi_feat)
    
    def project_rgb(self, rgb_feat):
        """
        RGB 特征投影
        
        【注意】调用时必须使用 torch.no_grad() 包裹
        
        Args:
            rgb_feat: [B, fusion_dim]
        Returns:
            z_rgb: [B, projector_dim]（L2 归一化）
        """
        return self.rgb_projector(rgb_feat)
    
    def classify(self, feat):
        """
        活动分类
        
        Args:
            feat: [B, fusion_dim]
        Returns:
            logits: [B, num_classes]
        """
        return self.classifier(feat)
    
    def regress(self, feat):
        """
        骨架回归
        
        Args:
            feat: [B, fusion_dim]
        Returns:
            joints: [B, num_joints * 3]
        """
        return self.regressor(feat)
    
    def get_trainable_params(self):
        """
        获取可训练参数（不包括 RGB Projector）
        
        用于创建 optimizer
        """
        params = []
        params.extend(self.csi_projector.parameters())
        params.extend(self.classifier.parameters())
        params.extend(self.regressor.parameters())
        return params
    
    def get_csi_projector_params(self):
        """获取 CSI Projector 参数（用于 EMA 更新 RGB Projector）"""
        return self.csi_projector.parameters()
    
    def get_rgb_projector_params(self):
        """获取 RGB Projector 参数"""
        return self.rgb_projector.parameters()


# 测试代码
if __name__ == '__main__':
    batch_size = 4
    fusion_dim = 256
    projector_dim = 128
    num_classes = 27
    num_joints = 17
    
    # 测试 Projector
    print("测试 Projector...")
    projector = Projector(fusion_dim, fusion_dim, projector_dim)
    x = torch.randn(batch_size, fusion_dim)
    z = projector(x)
    print(f"  输入: {x.shape}, 输出: {z.shape}")
    print(f"  L2 范数: {torch.norm(z, dim=1)}")  # 应该全为 1
    
    # 测试 ClassifierHead
    print("\n测试 ClassifierHead...")
    classifier = ClassifierHead(fusion_dim, 256, num_classes)
    logits = classifier(x)
    print(f"  输入: {x.shape}, 输出: {logits.shape}")
    
    # 测试 RegressorHead
    print("\n测试 RegressorHead...")
    regressor = RegressorHead(fusion_dim, 256, num_joints)
    joints = regressor(x)
    print(f"  输入: {x.shape}, 输出: {joints.shape}")
    
    # 测试 HeadsModule
    print("\n测试 HeadsModule...")
    heads = HeadsModule(
        fusion_dim=fusion_dim,
        projector_dim=projector_dim,
        classifier_hidden=256,
        regressor_hidden=256,
        num_classes=num_classes,
        num_joints=num_joints
    )
    
    z_csi = heads.project_csi(x)
    with torch.no_grad():
        z_rgb = heads.project_rgb(x)
    logits = heads.classify(x)
    joints = heads.regress(x)
    
    print(f"  CSI 投影: {z_csi.shape}")
    print(f"  RGB 投影: {z_rgb.shape}")
    print(f"  分类 logits: {logits.shape}")
    print(f"  骨架回归: {joints.shape}")
    
    # 验证 RGB Projector 参数已冻结
    print(f"\n  RGB Projector 参数已冻结: {all(not p.requires_grad for p in heads.rgb_projector.parameters())}")
