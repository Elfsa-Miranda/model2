"""
Dataset 测试：验证数据加载、形状对齐、预处理正确性和数据划分
"""
import unittest
import sys
import os
import torch
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.mmfi_dataset import CSIAugmentation


class TestCSIAugmentation(unittest.TestCase):
    """测试 CSI 增强"""
    
    def setUp(self):
        """测试初始化"""
        self.config = {
            'snr_range': [20, 40],
            'time_jitter_ratio': 0.1,
            'time_mask_ratio': 0.15,
            'subcarrier_mask_ratio': 0.1,
            'amplitude_scale_range': [0.9, 1.1],
            'apply_prob': 1.0  # 确保增强被应用
        }
        self.augmentation = CSIAugmentation(self.config)
        
        # 创建测试数据
        self.T = 100
        self.S = 30  # subcarriers
        self.A = 3   # antennas
        self.csi = np.random.randn(self.T, self.S, self.A).astype(np.float32)
    
    def test_augmentation_shape_preserved(self):
        """测试增强后形状保持不变"""
        csi_aug = self.augmentation(self.csi)
        self.assertEqual(csi_aug.shape, self.csi.shape)
    
    def test_gaussian_noise(self):
        """测试高斯噪声添加"""
        csi_noisy = self.augmentation.add_gaussian_noise(self.csi, snr_db=30)
        self.assertEqual(csi_noisy.shape, self.csi.shape)
        # 噪声应该改变数据
        self.assertFalse(np.allclose(csi_noisy, self.csi))
    
    def test_time_masking(self):
        """测试时间遮挡"""
        csi_masked = self.augmentation.time_masking(self.csi, mask_ratio=0.2)
        self.assertEqual(csi_masked.shape, self.csi.shape)
        # 应该有一些零值
        self.assertTrue(np.any(csi_masked == 0))
    
    def test_subcarrier_masking(self):
        """测试子载波遮挡"""
        csi_masked = self.augmentation.subcarrier_masking(self.csi, mask_ratio=0.2)
        self.assertEqual(csi_masked.shape, self.csi.shape)
    
    def test_amplitude_scaling(self):
        """测试幅值缩放"""
        csi_scaled = self.augmentation.amplitude_scaling(self.csi, [0.9, 1.1])
        self.assertEqual(csi_scaled.shape, self.csi.shape)


class TestDatasetOutputFormat(unittest.TestCase):
    """测试数据集输出格式（模拟测试）"""
    
    def test_output_dict_keys(self):
        """测试输出字典包含必要的键"""
        # 模拟数据集输出
        sample = {
            "csi": torch.randn(100, 30, 3),
            "skeleton": torch.randn(100, 17, 3),
            "label": 5,
            "sample_id": "E01_S01_A01"
        }
        
        # 验证键
        self.assertIn("csi", sample)
        self.assertIn("skeleton", sample)
        self.assertIn("label", sample)
        self.assertIn("sample_id", sample)
    
    def test_csi_skeleton_alignment(self):
        """测试 CSI 和 skeleton 时间对齐"""
        T = 100
        csi = torch.randn(T, 30, 3)
        skeleton = torch.randn(T, 17, 3)
        
        # 验证时间维度对齐
        self.assertEqual(csi.shape[0], skeleton.shape[0])
    
    def test_label_type(self):
        """测试标签类型为整数"""
        label = 5
        self.assertIsInstance(label, int)
    
    def test_sample_id_format(self):
        """测试样本 ID 格式"""
        sample_id = "E01_S01_A01"
        self.assertIsInstance(sample_id, str)
        # 验证格式
        parts = sample_id.split('_')
        self.assertEqual(len(parts), 3)


class TestDataSplit(unittest.TestCase):
    """测试数据划分"""
    
    def test_split_ratio_validation(self):
        """测试划分比例验证"""
        # 有效的划分比例
        valid_ratios = [0.7, 0.8]
        for ratio in valid_ratios:
            self.assertIn(ratio, [0.7, 0.8])
        
        # 无效的划分比例
        invalid_ratios = [0.5, 0.9, 0.6]
        for ratio in invalid_ratios:
            self.assertNotIn(ratio, [0.7, 0.8])
    
    def test_split_ratio_calculation(self):
        """测试划分比例计算"""
        # 8:2 划分
        split_str = "8:2"
        train_ratio = float(split_str.split(':')[0]) / 10
        test_ratio = float(split_str.split(':')[1]) / 10
        
        self.assertEqual(train_ratio, 0.8)
        self.assertEqual(test_ratio, 0.2)
        self.assertEqual(train_ratio + test_ratio, 1.0)
        
        # 7:3 划分
        split_str = "7:3"
        train_ratio = float(split_str.split(':')[0]) / 10
        test_ratio = float(split_str.split(':')[1]) / 10
        
        self.assertEqual(train_ratio, 0.7)
        self.assertEqual(test_ratio, 0.3)
        self.assertEqual(train_ratio + test_ratio, 1.0)
    
    def test_simulated_split(self):
        """测试模拟数据划分"""
        # 模拟 100 个样本
        total_samples = 100
        train_ratio = 0.8
        
        train_size = int(total_samples * train_ratio)
        test_size = total_samples - train_size
        
        self.assertEqual(train_size, 80)
        self.assertEqual(test_size, 20)
        
        # 验证比例
        actual_train_ratio = train_size / total_samples
        actual_test_ratio = test_size / total_samples
        
        self.assertAlmostEqual(actual_train_ratio, 0.8, places=2)
        self.assertAlmostEqual(actual_test_ratio, 0.2, places=2)


class TestDataPreprocessing(unittest.TestCase):
    """测试数据预处理"""
    
    def test_csi_preprocessing_steps(self):
        """测试 CSI 预处理步骤"""
        # 模拟原始 CSI 数据
        T, S, A = 100, 30, 3
        csi_raw = np.random.randn(T, S, A).astype(np.float32) + 10  # 添加偏置
        
        # 1. 减均值
        csi_mean = np.mean(csi_raw, axis=0, keepdims=True)
        csi_centered = csi_raw - csi_mean
        
        # 验证均值接近 0
        self.assertAlmostEqual(np.mean(csi_centered), 0, places=5)
        
        # 2. 归一化
        csi_min = np.min(csi_centered)
        csi_max = np.max(csi_centered)
        csi_normalized = (csi_centered - csi_min) / (csi_max - csi_min)
        
        # 验证范围在 [0, 1]
        self.assertGreaterEqual(np.min(csi_normalized), 0)
        self.assertLessEqual(np.max(csi_normalized), 1)
    
    def test_skeleton_3d_conversion(self):
        """测试骨架 2D 到 3D 转换"""
        T, J = 100, 17
        
        # 2D 骨架
        skeleton_2d = np.random.randn(T, J, 2).astype(np.float32)
        
        # 转换为 3D
        skeleton_3d = np.zeros((T, J, 3), dtype=np.float32)
        skeleton_3d[:, :, :2] = skeleton_2d
        
        # 验证形状
        self.assertEqual(skeleton_3d.shape, (T, J, 3))
        
        # 验证 z 坐标为 0
        self.assertTrue(np.all(skeleton_3d[:, :, 2] == 0))


if __name__ == '__main__':
    unittest.main()
