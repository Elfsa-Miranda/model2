"""
EMA 更新测试：验证指数移动平均更新公式的正确性
"""
import unittest
import sys
import os
import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.csi_encoder import CSIEncoder
from models.rgb_encoder import RGBEncoder
from models.momentum import momentum_update, MomentumUpdater, verify_ema_update


class TestEMAUpdate(unittest.TestCase):
    """测试 EMA 更新"""
    
    def setUp(self):
        """测试初始化"""
        # 使用简单的线性层进行测试
        self.csi_encoder = torch.nn.Linear(10, 5)
        self.rgb_encoder = torch.nn.Linear(10, 5)
        
        # 冻结 RGB 参数
        for p in self.rgb_encoder.parameters():
            p.requires_grad = False
    
    def test_ema_formula_correctness(self):
        """测试 EMA 更新公式数值正确性"""
        m = 0.999
        
        # 保存更新前的参数
        old_rgb_weight = self.rgb_encoder.weight.data.clone()
        old_rgb_bias = self.rgb_encoder.bias.data.clone()
        csi_weight = self.csi_encoder.weight.data.clone()
        csi_bias = self.csi_encoder.bias.data.clone()
        
        # 执行 EMA 更新
        momentum_update(self.rgb_encoder, self.csi_encoder, m)
        
        # 计算期望值
        expected_weight = m * old_rgb_weight + (1 - m) * csi_weight
        expected_bias = m * old_rgb_bias + (1 - m) * csi_bias
        
        # 验证
        self.assertTrue(
            torch.allclose(self.rgb_encoder.weight.data, expected_weight, atol=1e-6),
            "EMA 更新公式不正确（weight）"
        )
        self.assertTrue(
            torch.allclose(self.rgb_encoder.bias.data, expected_bias, atol=1e-6),
            "EMA 更新公式不正确（bias）"
        )
    
    def test_ema_momentum_range(self):
        """测试动量参数在合理范围内"""
        # 有效的动量值
        valid_momentums = [0.9, 0.99, 0.999, 0.9999]
        for m in valid_momentums:
            # 不应该抛出异常
            momentum_update(self.rgb_encoder, self.csi_encoder, m)
        
        # 无效的动量值
        invalid_momentums = [0, 1, -0.1, 1.1]
        for m in invalid_momentums:
            with self.assertRaises(AssertionError):
                momentum_update(self.rgb_encoder, self.csi_encoder, m)
    
    def test_parameter_shape_consistency(self):
        """测试 EMA 更新后参数形状保持一致"""
        original_shapes = [p.shape for p in self.rgb_encoder.parameters()]
        
        momentum_update(self.rgb_encoder, self.csi_encoder, 0.999)
        
        new_shapes = [p.shape for p in self.rgb_encoder.parameters()]
        
        self.assertEqual(original_shapes, new_shapes)
    
    def test_rgb_requires_grad_assertion(self):
        """测试 RGB 参数必须被冻结"""
        # 解冻 RGB 参数
        for p in self.rgb_encoder.parameters():
            p.requires_grad = True
        
        # 应该抛出异常
        with self.assertRaises(AssertionError):
            momentum_update(self.rgb_encoder, self.csi_encoder, 0.999)
        
        # 重新冻结
        for p in self.rgb_encoder.parameters():
            p.requires_grad = False
    
    def test_verify_ema_update_function(self):
        """测试 verify_ema_update 辅助函数"""
        result = verify_ema_update(self.rgb_encoder, self.csi_encoder, m=0.999)
        self.assertTrue(result)


class TestMomentumUpdater(unittest.TestCase):
    """测试 MomentumUpdater 类"""
    
    def setUp(self):
        """测试初始化"""
        self.csi_encoder = torch.nn.Linear(10, 5)
        self.rgb_encoder = torch.nn.Linear(10, 5)
        
        for p in self.rgb_encoder.parameters():
            p.requires_grad = False
        
        self.updater = MomentumUpdater(
            self.rgb_encoder, 
            self.csi_encoder, 
            m=0.999
        )
    
    def test_updater_initialization(self):
        """测试更新器初始化"""
        self.assertEqual(self.updater.base_m, 0.999)
        self.assertEqual(self.updater.current_step, 0)
    
    def test_updater_update(self):
        """测试更新器更新"""
        old_weight = self.rgb_encoder.weight.data.clone()
        
        self.updater.update()
        
        # 参数应该改变
        self.assertFalse(torch.allclose(self.rgb_encoder.weight.data, old_weight))
        self.assertEqual(self.updater.current_step, 1)
    
    def test_updater_state_dict(self):
        """测试状态保存和加载"""
        self.updater.update()
        self.updater.update()
        
        state = self.updater.state_dict()
        
        self.assertEqual(state['current_step'], 2)
        
        # 创建新的更新器并加载状态
        new_updater = MomentumUpdater(
            self.rgb_encoder, 
            self.csi_encoder, 
            m=0.999
        )
        new_updater.load_state_dict(state)
        
        self.assertEqual(new_updater.current_step, 2)


if __name__ == '__main__':
    unittest.main()
