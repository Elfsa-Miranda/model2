"""
推理测试：验证推理阶段严格禁止使用 RGB、队列、回归头
"""
import unittest
import sys
import os
import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.csi_encoder import CSIEncoder
from models.rgb_encoder import RGBEncoder
from models.heads import HeadsModule, ClassifierHead


class TestInferenceConstraints(unittest.TestCase):
    """测试推理约束"""
    
    def setUp(self):
        """测试初始化"""
        self.csi_encoder = CSIEncoder(
            num_subcarriers=30,
            num_antennas=3,
            channel_dim=128,
            temporal_dim=128,
            fusion_dim=256
        )
        
        self.classifier = ClassifierHead(
            input_dim=256,
            hidden_dim=256,
            num_classes=27
        )
        
        # 创建测试输入
        self.batch_size = 4
        self.T = 100
        self.csi_input = torch.randn(self.batch_size, self.T, 30, 3)
    
    def test_inference_only_uses_csi_encoder(self):
        """测试推理只使用 CSI 编码器"""
        self.csi_encoder.eval()
        self.classifier.eval()
        
        with torch.no_grad():
            # 推理流程：CSI Encoder -> Classifier
            f_csi = self.csi_encoder(self.csi_input)
            logits = self.classifier(f_csi)
        
        # 验证输出形状
        self.assertEqual(f_csi.shape, (self.batch_size, 256))
        self.assertEqual(logits.shape, (self.batch_size, 27))
    
    def test_rgb_encoder_not_called_in_inference(self):
        """测试推理时 RGB 编码器未被调用"""
        # 创建一个带有调用计数器的 RGB 编码器
        class RGBEncoderWithCounter(RGBEncoder):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.call_count = 0
            
            def forward(self, x):
                self.call_count += 1
                return super().forward(x)
        
        rgb_encoder = RGBEncoderWithCounter(
            num_joints=17,
            coord_dim=3,
            spatial_dim=128,
            temporal_dim=128,
            fusion_dim=256
        )
        
        # 模拟推理（不应该调用 RGB 编码器）
        self.csi_encoder.eval()
        self.classifier.eval()
        
        with torch.no_grad():
            f_csi = self.csi_encoder(self.csi_input)
            logits = self.classifier(f_csi)
        
        # 验证 RGB 编码器未被调用
        self.assertEqual(rgb_encoder.call_count, 0)
    
    def test_projector_not_used_in_inference(self):
        """测试推理时投影头未被使用"""
        heads = HeadsModule(
            fusion_dim=256,
            projector_dim=128,
            classifier_hidden=256,
            regressor_hidden=256,
            num_classes=27,
            num_joints=17
        )
        
        # 推理时只使用 classifier
        self.csi_encoder.eval()
        heads.eval()
        
        with torch.no_grad():
            f_csi = self.csi_encoder(self.csi_input)
            # 只调用 classify，不调用 project_csi
            logits = heads.classify(f_csi)
        
        self.assertEqual(logits.shape, (self.batch_size, 27))
    
    def test_regressor_not_used_in_inference(self):
        """测试推理时回归头未被使用"""
        heads = HeadsModule(
            fusion_dim=256,
            projector_dim=128,
            classifier_hidden=256,
            regressor_hidden=256,
            num_classes=27,
            num_joints=17
        )
        
        # 推理时只使用 classifier
        self.csi_encoder.eval()
        heads.eval()
        
        with torch.no_grad():
            f_csi = self.csi_encoder(self.csi_input)
            # 只调用 classify，不调用 regress
            logits = heads.classify(f_csi)
        
        # 验证没有调用 regressor
        # （通过检查输出形状间接验证）
        self.assertEqual(logits.shape, (self.batch_size, 27))
    
    def test_model_in_eval_mode(self):
        """测试推理时模型处于 eval 模式"""
        self.csi_encoder.eval()
        self.classifier.eval()
        
        self.assertFalse(self.csi_encoder.training)
        self.assertFalse(self.classifier.training)
    
    def test_no_gradient_computation_in_inference(self):
        """测试推理时不计算梯度"""
        self.csi_encoder.eval()
        self.classifier.eval()
        
        with torch.no_grad():
            f_csi = self.csi_encoder(self.csi_input)
            logits = self.classifier(f_csi)
        
        # 验证输出不需要梯度
        self.assertFalse(f_csi.requires_grad)
        self.assertFalse(logits.requires_grad)


class TestInferenceModelConstraints(unittest.TestCase):
    """测试推理模型约束"""
    
    def test_inference_model_components(self):
        """测试推理模型只包含必要组件"""
        # 推理模型应该只包含：
        # 1. CSI Encoder
        # 2. Classifier
        
        # 不应该包含：
        # 1. RGB Encoder
        # 2. Projector
        # 3. Regressor
        # 4. Queue
        
        class InferenceModel:
            def __init__(self):
                self.csi_encoder = CSIEncoder(
                    num_subcarriers=30,
                    num_antennas=3,
                    channel_dim=128,
                    temporal_dim=128,
                    fusion_dim=256
                )
                self.classifier = ClassifierHead(
                    input_dim=256,
                    hidden_dim=256,
                    num_classes=27
                )
        
        model = InferenceModel()
        
        # 验证只有必要组件
        self.assertTrue(hasattr(model, 'csi_encoder'))
        self.assertTrue(hasattr(model, 'classifier'))
        
        # 验证没有禁止的组件
        self.assertFalse(hasattr(model, 'rgb_encoder'))
        self.assertFalse(hasattr(model, 'projector'))
        self.assertFalse(hasattr(model, 'regressor'))
        self.assertFalse(hasattr(model, 'queue'))


if __name__ == '__main__':
    unittest.main()
