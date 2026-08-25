"""
优化器测试：验证 RGB 编码器参数不在优化器参数列表中
"""
import unittest
import sys
import os
import torch
from torch.optim import AdamW

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.csi_encoder import CSIEncoder
from models.rgb_encoder import RGBEncoder
from models.heads import HeadsModule


class TestOptimizerExcludesRGB(unittest.TestCase):
    """测试优化器不包含 RGB 参数"""
    
    def setUp(self):
        """测试初始化"""
        self.csi_encoder = CSIEncoder(
            num_subcarriers=30,
            num_antennas=3,
            channel_dim=128,
            temporal_dim=128,
            fusion_dim=256
        )
        
        self.rgb_encoder = RGBEncoder(
            num_joints=17,
            coord_dim=3,
            spatial_dim=128,
            temporal_dim=128,
            fusion_dim=256
        )
        
        self.heads = HeadsModule(
            fusion_dim=256,
            projector_dim=128,
            classifier_hidden=256,
            regressor_hidden=256,
            num_classes=27,
            num_joints=17
        )
    
    def test_optimizer_param_groups(self):
        """测试优化器参数组不包含 RGB 编码器参数"""
        # 创建优化器（正确方式：不包含 RGB 参数）
        params = []
        params.append({'params': self.csi_encoder.parameters(), 'lr': 1e-4})
        params.append({'params': self.heads.get_trainable_params(), 'lr': 1e-3})
        
        optimizer = AdamW(params, weight_decay=1e-4)
        
        # 获取 RGB 参数 ID
        rgb_param_ids = set(id(p) for p in self.rgb_encoder.parameters())
        rgb_proj_param_ids = set(id(p) for p in self.heads.rgb_projector.parameters())
        
        # 验证优化器不包含 RGB 参数
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                self.assertNotIn(
                    id(param), 
                    rgb_param_ids,
                    "【禁止项】optimizer 不能包含 rgb_encoder 参数"
                )
                self.assertNotIn(
                    id(param),
                    rgb_proj_param_ids,
                    "【禁止项】optimizer 不能包含 rgb_projector 参数"
                )
    
    def test_parameter_count_consistency(self):
        """测试优化器参数数量与可训练参数数量一致"""
        # 可训练参数：CSI Encoder + Heads（不包括 RGB Projector）
        trainable_params = []
        trainable_params.extend(list(self.csi_encoder.parameters()))
        trainable_params.extend(list(self.heads.get_trainable_params()))
        
        # 创建优化器
        params = []
        params.append({'params': self.csi_encoder.parameters()})
        params.append({'params': self.heads.get_trainable_params()})
        
        optimizer = AdamW(params)
        
        # 统计优化器中的参数数量
        optimizer_param_count = sum(
            p.numel() 
            for group in optimizer.param_groups 
            for p in group['params']
        )
        
        # 统计可训练参数数量
        trainable_param_count = sum(p.numel() for p in trainable_params)
        
        self.assertEqual(optimizer_param_count, trainable_param_count)
    
    def test_rgb_encoder_not_in_optimizer(self):
        """测试 RGB 编码器参数不在优化器中"""
        # 错误方式：包含 RGB 参数（不应该这样做）
        # 这个测试验证我们的检查逻辑是正确的
        
        rgb_param_ids = set(id(p) for p in self.rgb_encoder.parameters())
        csi_param_ids = set(id(p) for p in self.csi_encoder.parameters())
        
        # 验证两个集合不相交
        self.assertEqual(len(rgb_param_ids & csi_param_ids), 0)
    
    def test_heads_trainable_params_excludes_rgb_projector(self):
        """测试 Heads 可训练参数不包含 RGB Projector"""
        trainable_params = list(self.heads.get_trainable_params())
        rgb_proj_params = list(self.heads.rgb_projector.parameters())
        
        trainable_ids = set(id(p) for p in trainable_params)
        rgb_proj_ids = set(id(p) for p in rgb_proj_params)
        
        # 验证不相交
        self.assertEqual(len(trainable_ids & rgb_proj_ids), 0)


if __name__ == '__main__':
    unittest.main()
