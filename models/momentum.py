"""
Exponential Moving Average (EMA) Momentum Update

核心机制：
- RGB Encoder 的参数通过 CSI Encoder 参数的 EMA 更新
- 更新公式: θ_rgb ← m * θ_rgb + (1-m) * θ_csi
- m 为动量因子，默认 0.999

关键约束：
1. RGB Encoder 参数不参与梯度反向传播
2. EMA 更新在 optimizer.step() 之后进行
3. 更新时使用 torch.no_grad()

物理意义：
- 保持 RGB Encoder 的语义稳定性
- 避免 RGB 表征空间剧烈变化
- 提供稳定的语义锚点用于对比学习
"""

import torch
import torch.nn as nn
from typing import Iterator, Tuple


@torch.no_grad()
def momentum_update(rgb_encoder: nn.Module, 
                    csi_encoder: nn.Module, 
                    m: float = 0.999) -> None:
    """
    执行 EMA 动量更新
    
    更新公式: θ_rgb ← m * θ_rgb + (1-m) * θ_csi
    
    Args:
        rgb_encoder: RGB Encoder（Key Encoder）
        csi_encoder: CSI Encoder（Query Encoder）
        m: 动量因子，范围 (0, 1)，默认 0.999
    
    【关键约束】
    - 此函数必须在 optimizer.step() 之后调用
    - 使用 @torch.no_grad() 装饰器确保不建立计算图
    """
    # 验证动量因子范围
    assert 0 < m < 1, f"动量因子 m 必须在 (0, 1) 范围内，当前值: {m}"
    
    # 验证 RGB Encoder 参数已冻结
    assert all(not p.requires_grad for p in rgb_encoder.parameters()), \
        "RGB Encoder 参数必须被冻结（requires_grad=False）"
    
    # 执行 EMA 更新
    for p_rgb, p_csi in zip(rgb_encoder.parameters(), csi_encoder.parameters()):
        # θ_rgb ← m * θ_rgb + (1-m) * θ_csi
        p_rgb.data.mul_(m).add_(p_csi.data, alpha=1 - m)


@torch.no_grad()
def copy_params(target_encoder: nn.Module, 
                source_encoder: nn.Module) -> None:
    """
    将源编码器的参数复制到目标编码器
    
    用于初始化时将 CSI Encoder 参数复制到 RGB Encoder
    
    Args:
        target_encoder: 目标编码器（RGB Encoder）
        source_encoder: 源编码器（CSI Encoder）
    """
    for p_target, p_source in zip(target_encoder.parameters(), source_encoder.parameters()):
        p_target.data.copy_(p_source.data)


class MomentumUpdater:
    """
    动量更新器类
    
    封装 EMA 更新逻辑，提供更方便的接口
    """
    
    def __init__(self, 
                 rgb_encoder: nn.Module,
                 csi_encoder: nn.Module,
                 m: float = 0.999,
                 warmup_steps: int = 0):
        """
        Args:
            rgb_encoder: RGB Encoder（Key Encoder）
            csi_encoder: CSI Encoder（Query Encoder）
            m: 动量因子，默认 0.999
            warmup_steps: 预热步数，在此期间使用较小的动量
        """
        self.rgb_encoder = rgb_encoder
        self.csi_encoder = csi_encoder
        self.base_m = m
        self.m = m
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
        # 验证 RGB Encoder 参数已冻结
        self._verify_frozen()
    
    def _verify_frozen(self):
        """验证 RGB Encoder 参数已冻结"""
        assert all(not p.requires_grad for p in self.rgb_encoder.parameters()), \
            "RGB Encoder 参数必须被冻结（requires_grad=False）"
    
    def update(self):
        """
        执行一次 EMA 更新
        
        【关键】必须在 optimizer.step() 之后调用
        """
        # 更新动量（可选的预热策略）
        if self.current_step < self.warmup_steps:
            # 预热期间使用较小的动量，让 RGB Encoder 更快跟随 CSI Encoder
            self.m = self.base_m * (self.current_step / self.warmup_steps)
        else:
            self.m = self.base_m
        
        # 执行 EMA 更新
        momentum_update(self.rgb_encoder, self.csi_encoder, self.m)
        
        self.current_step += 1
    
    def initialize_from_csi(self):
        """
        从 CSI Encoder 初始化 RGB Encoder 参数
        
        在训练开始前调用
        """
        copy_params(self.rgb_encoder, self.csi_encoder)
    
    def get_current_momentum(self) -> float:
        """获取当前动量值"""
        return self.m
    
    def state_dict(self) -> dict:
        """保存状态"""
        return {
            'current_step': self.current_step,
            'm': self.m
        }
    
    def load_state_dict(self, state_dict: dict):
        """加载状态"""
        self.current_step = state_dict['current_step']
        self.m = state_dict['m']


def verify_ema_update(rgb_encoder: nn.Module,
                      csi_encoder: nn.Module,
                      m: float = 0.999,
                      tolerance: float = 1e-6) -> bool:
    """
    验证 EMA 更新公式的正确性
    
    用于单元测试
    
    Args:
        rgb_encoder: RGB Encoder
        csi_encoder: CSI Encoder
        m: 动量因子
        tolerance: 数值容差
    
    Returns:
        bool: 验证是否通过
    """
    # 保存更新前的 RGB 参数
    old_params = [p.data.clone() for p in rgb_encoder.parameters()]
    csi_params = [p.data.clone() for p in csi_encoder.parameters()]
    
    # 执行 EMA 更新
    momentum_update(rgb_encoder, csi_encoder, m)
    
    # 验证更新后的参数
    for old_p, csi_p, new_p in zip(old_params, csi_params, rgb_encoder.parameters()):
        expected = m * old_p + (1 - m) * csi_p
        if not torch.allclose(new_p.data, expected, atol=tolerance):
            return False
    
    return True


# 测试代码
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '..')
    from models.csi_encoder import CSIEncoder
    from models.rgb_encoder import RGBEncoder
    
    # 创建编码器
    csi_encoder = CSIEncoder(
        num_subcarriers=30,
        num_antennas=3,
        channel_dim=128,
        temporal_dim=128,
        fusion_dim=256
    )
    
    rgb_encoder = RGBEncoder(
        num_joints=17,
        coord_dim=3,
        spatial_dim=128,
        temporal_dim=128,
        fusion_dim=256
    )
    
    # 验证 RGB Encoder 参数已冻结
    print(f"RGB Encoder 参数已冻结: {all(not p.requires_grad for p in rgb_encoder.parameters())}")
    
    # 测试 EMA 更新
    m = 0.999
    print(f"\n测试 EMA 更新 (m={m})...")
    
    # 保存更新前的参数
    old_rgb_param = list(rgb_encoder.parameters())[0].data.clone()
    csi_param = list(csi_encoder.parameters())[0].data.clone()
    
    # 执行更新
    momentum_update(rgb_encoder, csi_encoder, m)
    
    # 验证更新
    new_rgb_param = list(rgb_encoder.parameters())[0].data
    expected = m * old_rgb_param + (1 - m) * csi_param
    
    print(f"更新前 RGB 参数均值: {old_rgb_param.mean().item():.6f}")
    print(f"CSI 参数均值: {csi_param.mean().item():.6f}")
    print(f"更新后 RGB 参数均值: {new_rgb_param.mean().item():.6f}")
    print(f"期望值均值: {expected.mean().item():.6f}")
    print(f"验证通过: {torch.allclose(new_rgb_param, expected)}")
    
    # 测试 MomentumUpdater
    print("\n测试 MomentumUpdater...")
    updater = MomentumUpdater(rgb_encoder, csi_encoder, m=0.999)
    updater.update()
    print(f"当前动量: {updater.get_current_momentum()}")
