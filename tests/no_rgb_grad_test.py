"""
RGB 梯度测试：验证 RGB 编码器参数被正确冻结
"""
import unittest
import sys
import os
import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rgb_encoder import RGBEncoder


class TestRGBGradientFrozen(unittest.TestCase):
    """测试 RGB 编码器梯度冻结"""
    
    def setUp(self):
        """测试初始化"""
        self.rgb_encoder = RGBEncoder(
            num_joints=17,
            coord_dim=3,
            spatial_dim=128,
            temporal_dim=128,
            fusion_dim=256
        )
    
    def test_rgb_parameters_frozen(self):
        """测试 RGB 编码器所有参数的 requires_grad 为 False"""
        # 【关键断言】所有参数必须被冻结
        self.assertTrue(
            all(not p.requires_grad for p in self.rgb_encoder.parameters()),
            "RGB Encoder 所有参数的 requires_grad 必须为 False"
        )
    
    def test_rgb_no_grad_context(self):
        """测试 RGB 前向传播在 no_grad 上下文中执行"""
        batch_size = 4
        T = 100
        num_joints = 17
        
        x = torch.randn(batch_size, T, num_joints, 3)
        
        # 在 no_grad 上下文中执行
        with torch.no_grad():
            output = self.rgb_encoder(x)
        
        # 验证输出没有梯度
        self.assertFalse(output.requires_grad)
    
    def test_rgb_forward_no_grad_required(self):
        """测试 RGB 前向传播不建立计算图"""
        batch_size = 4
        T = 100
        num_joints = 17
        
        x = torch.randn(batch_size, T, num_joints, 3)
        
        # 即使不使用 no_grad，由于参数被冻结，也不应该建立计算图
        output = self.rgb_encoder(x)
        
        # 由于所有参数都被冻结，输出不应该需要梯度
        # 注意：这取决于输入是否需要梯度
        # 如果输入不需要梯度，输出也不需要
        self.assertFalse(output.requires_grad)
    
    def test_parameter_count(self):
        """测试参数数量"""
        total_params = sum(p.numel() for p in self.rgb_encoder.parameters())
        frozen_params = sum(p.numel() for p in self.rgb_encoder.parameters() if not p.requires_grad)
        
        # 所有参数都应该被冻结
        self.assertEqual(total_params, frozen_params)


if __name__ == '__main__':
    unittest.main()
