"""
MMFi Dataset Wrapper for Cross-modal Momentum Contrastive Learning

数据对齐策略：
- CSI 和 RGB（骨架）数据通过帧索引严格对齐
- 每个样本确保 csi[t] 对应 skeleton[t]
- 使用官方 MMFi loader 读取原始数据，在 wrapper 中进行预处理

数据划分：
- 默认使用 8:2 比例划分训练集和测试集（导师要求）
- 支持 7:3 和 8:2 两种划分比例
- 使用随机种子确保可复现性

CSI 预处理（保持动作语义）：
- 仅使用幅值（abs），避免相位不稳定
- Per-subcarrier 减均值（去除环境偏置）
- 简单时间平滑（moving average）减少噪声
- Instance-wise 归一化

CSI 增强（来自 DeepCRF，保持动作语义）：
- 加性高斯噪声（SNR ∈ [20, 40] dB）
- 时间轴 jitter/shift（±5-10% window）
- Time masking（随机遮挡连续时间段）
- Subcarrier masking（随机遮挡部分子载波）
- 幅值 scaling（0.9-1.1）
- 禁止：打乱时间顺序、破坏子载波结构
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加 MMFi 官方库路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'MMFi_dataset'))


class CSIAugmentation:
    """
    CSI 专用增强（DeepCRF 风格）
    关键原则：只对不会破坏人体动作语义的 CSI 维度做增强
    """
    
    def __init__(self, config):
        self.snr_range = config.get('snr_range', [20, 40])  # dB
        self.time_jitter_ratio = config.get('time_jitter_ratio', 0.1)  # ±10%
        self.time_mask_ratio = config.get('time_mask_ratio', 0.15)
        self.subcarrier_mask_ratio = config.get('subcarrier_mask_ratio', 0.1)
        self.amplitude_scale_range = config.get('amplitude_scale_range', [0.9, 1.1])
        self.apply_prob = config.get('apply_prob', 0.8)
    
    def add_gaussian_noise(self, csi, snr_db):
        """
        添加高斯噪声（指定 SNR）
        Args:
            csi: [T, Subcarriers, Antennas]
            snr_db: 信噪比（dB）
        """
        signal_power = np.mean(csi ** 2)
        if signal_power < 1e-10:
            return csi
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise = np.random.normal(0, np.sqrt(noise_power), csi.shape)
        return csi + noise
    
    def time_jitter(self, csi, ratio):
        """
        时间轴 jitter（轻微时间偏移）
        Args:
            csi: [T, Subcarriers, Antennas]
            ratio: 偏移比例
        """
        T = csi.shape[0]
        shift = int(T * ratio * (2 * np.random.random() - 1))  # [-ratio*T, +ratio*T]
        if shift > 0:
            return np.concatenate([csi[shift:], csi[-shift:]], axis=0)
        elif shift < 0:
            return np.concatenate([csi[:shift], csi[:-shift]], axis=0)
        return csi
    
    def time_masking(self, csi, mask_ratio):
        """
        时间遮挡（随机遮挡连续时间段）
        Args:
            csi: [T, Subcarriers, Antennas]
            mask_ratio: 遮挡比例
        """
        T = csi.shape[0]
        mask_length = int(T * mask_ratio)
        if mask_length == 0:
            return csi
        start = np.random.randint(0, max(1, T - mask_length + 1))
        csi_masked = csi.copy()
        csi_masked[start:start+mask_length] = 0
        return csi_masked
    
    def subcarrier_masking(self, csi, mask_ratio):
        """
        子载波遮挡（随机遮挡部分子载波）
        Args:
            csi: [T, Subcarriers, Antennas]
            mask_ratio: 遮挡比例
        """
        num_subcarriers = csi.shape[1]
        num_mask = int(num_subcarriers * mask_ratio)
        if num_mask == 0:
            return csi
        mask_indices = np.random.choice(num_subcarriers, num_mask, replace=False)
        csi_masked = csi.copy()
        csi_masked[:, mask_indices, :] = 0
        return csi_masked
    
    def amplitude_scaling(self, csi, scale_range):
        """
        幅值缩放
        Args:
            csi: [T, Subcarriers, Antennas]
            scale_range: [min_scale, max_scale]
        """
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return csi * scale
    
    def __call__(self, csi):
        """
        应用随机增强组合
        Args:
            csi: [T, Subcarriers, Antennas]
        Returns:
            augmented_csi: [T, Subcarriers, Antennas]
        """
        if np.random.random() > self.apply_prob:
            return csi
        
        csi_aug = csi.copy()
        
        # 随机应用各种增强
        if np.random.random() > 0.5:
            snr = np.random.uniform(self.snr_range[0], self.snr_range[1])
            csi_aug = self.add_gaussian_noise(csi_aug, snr)
        
        if np.random.random() > 0.5:
            csi_aug = self.time_jitter(csi_aug, self.time_jitter_ratio)
        
        if np.random.random() > 0.5:
            csi_aug = self.time_masking(csi_aug, self.time_mask_ratio)
        
        if np.random.random() > 0.5:
            csi_aug = self.subcarrier_masking(csi_aug, self.subcarrier_mask_ratio)
        
        if np.random.random() > 0.5:
            csi_aug = self.amplitude_scaling(csi_aug, self.amplitude_scale_range)
        
        return csi_aug


class MMFiDatasetWrapper(Dataset):
    """
    MMFi 数据集包装器
    
    功能：
    1. 使用官方 loader 读取 CSI 和 RGB（骨架）数据
    2. CSI 预处理：幅值提取、减均值、平滑、归一化
    3. 严格对齐：确保 csi[t] 对应 skeleton[t]
    4. 支持训练/验证/测试划分（默认 8:2）
    
    返回格式：
    {
        "csi": torch.FloatTensor([T, Subcarriers, Antennas]),
        "skeleton": torch.FloatTensor([T, J, 3]),
        "label": int,
        "sample_id": str
    }
    """
    
    def __init__(self, 
                 data_root,
                 split_mode='train',
                 split_ratio=0.8,  # 默认 8:2 划分
                 protocol='protocol2',
                 data_unit='sequence',
                 smooth_kernel_size=3,
                 augmentation_config=None,
                 random_seed=0):
        """
        Args:
            data_root: MMFi 数据集根目录
            split_mode: 'train', 'val', 'test'
            split_ratio: 训练集比例（0.7 或 0.8），默认 0.8（8:2 划分）
            protocol: 'protocol1' (daily) 或 'protocol2' (rehabilitation)
            data_unit: 'sequence' 或 'frame'
            smooth_kernel_size: 移动平均核大小
            augmentation_config: CSI 增强配置（仅训练时使用）
            random_seed: 随机种子
        """
        self.data_root = data_root
        self.split_mode = split_mode
        self.split_ratio = split_ratio
        self.protocol = protocol
        self.data_unit = data_unit
        self.smooth_kernel_size = smooth_kernel_size
        self.random_seed = random_seed
        
        # 验证划分比例
        assert split_ratio in [0.7, 0.8], f"split_ratio 必须是 0.7 或 0.8，当前值: {split_ratio}"
        
        # CSI 增强（仅训练时）
        self.augmentation = None
        if split_mode == 'train' and augmentation_config is not None:
            self.augmentation = CSIAugmentation(augmentation_config)
        
        # 加载官方数据集
        self._load_official_dataset()
        
        # 动作标签映射
        self.action_to_label = self._build_action_mapping()
        
        # 打印数据集信息
        self._print_dataset_info()
    
    def _load_official_dataset(self):
        """使用官方 loader 加载数据"""
        from mmfi_lib.mmfi import MMFi_Database, MMFi_Dataset as MMFi_Dataset_Official, decode_config
        
        # 构建配置
        config = {
            'modality': 'wifi-csi|rgb',  # 同时加载 CSI 和 RGB
            'protocol': self.protocol,
            'data_unit': self.data_unit,
            'split_to_use': 'random_split',
            'random_split': {
                'ratio': self.split_ratio,
                'random_seed': self.random_seed
            }
        }
        
        # 创建数据库
        database = MMFi_Database(self.data_root)
        
        # 解码配置
        config_dataset = decode_config(config)
        
        # 根据 split_mode 选择数据集
        if self.split_mode in ['train', 'training']:
            dataset_config = config_dataset['train_dataset']
        else:
            dataset_config = config_dataset['val_dataset']
        
        # 创建官方数据集
        self.official_dataset = MMFi_Dataset_Official(
            database, 
            self.data_unit,
            **dataset_config
        )
    
    def _build_action_mapping(self):
        """构建动作到标签的映射"""
        if self.protocol == 'protocol1':
            actions = ['A02', 'A03', 'A04', 'A05', 'A13', 'A14', 'A17', 'A18', 
                      'A19', 'A20', 'A21', 'A22', 'A23', 'A27']
        elif self.protocol == 'protocol2':
            actions = ['A01', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 
                      'A15', 'A16', 'A24', 'A25', 'A26']
        else:
            actions = [f'A{i:02d}' for i in range(1, 28)]
        
        return {action: idx for idx, action in enumerate(sorted(actions))}
    
    def _print_dataset_info(self):
        """打印数据集信息"""
        logger.info(f"=" * 50)
        logger.info(f"MMFi 数据集加载完成")
        logger.info(f"  划分模式: {self.split_mode}")
        logger.info(f"  划分比例: {self.split_ratio:.0%} 训练 / {1-self.split_ratio:.0%} 测试")
        logger.info(f"  协议: {self.protocol}")
        logger.info(f"  数据单元: {self.data_unit}")
        logger.info(f"  样本数量: {len(self.official_dataset)}")
        logger.info(f"  类别数量: {len(self.action_to_label)}")
        logger.info(f"  随机种子: {self.random_seed}")
        logger.info(f"  CSI 增强: {'启用' if self.augmentation else '禁用'}")
        logger.info(f"=" * 50)
    
    def _preprocess_csi(self, csi_raw):
        """
        CSI 预处理
        Args:
            csi_raw: [T, Antennas, Subcarriers, Features] 原始 CSI 数据
        Returns:
            csi_processed: [T, Subcarriers, Antennas] 预处理后的 CSI
        """
        # 检查输入形状
        if len(csi_raw.shape) == 4:
            T, A, S, F = csi_raw.shape
            # 取幅值（通常是第一个特征或计算模长）
            if F >= 2:
                # 假设前两个特征是实部和虚部，计算幅值
                csi_amplitude = np.sqrt(csi_raw[:, :, :, 0]**2 + csi_raw[:, :, :, 1]**2)
            else:
                csi_amplitude = csi_raw[:, :, :, 0]
            
            # 重新排列为 [T, Subcarriers, Antennas]
            csi = csi_amplitude.transpose(0, 2, 1).astype(np.float32)
        elif len(csi_raw.shape) == 3:
            # 已经是 [T, Subcarriers, Antennas] 格式
            csi = csi_raw.copy().astype(np.float32)
        else:
            raise ValueError(f"不支持的 CSI 形状: {csi_raw.shape}")
        
        # 2. Per-subcarrier 减均值（去除环境偏置）
        csi_mean = np.mean(csi, axis=0, keepdims=True)  # [1, Subcarriers, Antennas]
        csi = csi - csi_mean
        
        # 3. 时间平滑（moving average）
        if self.smooth_kernel_size > 1:
            kernel = np.ones(self.smooth_kernel_size) / self.smooth_kernel_size
            T, S, A = csi.shape
            csi_smoothed = np.zeros_like(csi)
            for s in range(S):
                for a in range(A):
                    csi_smoothed[:, s, a] = np.convolve(csi[:, s, a], kernel, mode='same')
            csi = csi_smoothed
        
        # 4. Instance-wise 归一化
        csi_min = np.min(csi)
        csi_max = np.max(csi)
        if csi_max - csi_min > 1e-6:
            csi = (csi - csi_min) / (csi_max - csi_min)
        
        return csi
    
    def __len__(self):
        return len(self.official_dataset)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        Returns:
            {
                "csi": torch.FloatTensor([T, Subcarriers, Antennas]),
                "skeleton": torch.FloatTensor([T, J, 3]),
                "label": int,
                "sample_id": str
            }
        """
        # 从官方数据集获取数据
        sample = self.official_dataset[idx]
        
        # 提取 CSI 数据
        csi_raw = sample['input_wifi-csi']  # [T, Subcarriers, Antennas]
        
        # 提取 RGB 骨架数据
        skeleton_raw = sample['input_rgb']  # [T, J, 3] 或 [T, J, 2]
        
        # 确保骨架是 3D（如果是 2D，补零）
        if len(skeleton_raw.shape) == 2:
            # 单帧情况
            skeleton_raw = skeleton_raw.reshape(1, -1, skeleton_raw.shape[-1])
        
        if skeleton_raw.shape[-1] == 2:
            T, J, _ = skeleton_raw.shape
            skeleton_3d = np.zeros((T, J, 3), dtype=np.float32)
            skeleton_3d[:, :, :2] = skeleton_raw
            skeleton_raw = skeleton_3d
        
        # CSI 预处理
        csi_processed = self._preprocess_csi(csi_raw)
        
        # CSI 增强（仅训练时）
        if self.augmentation is not None:
            csi_processed = self.augmentation(csi_processed)
        
        # 动作标签
        action = sample['action']
        label = self.action_to_label[action]
        
        # 样本 ID
        sample_id = f"{sample['scene']}_{sample['subject']}_{action}"
        if 'idx' in sample:
            sample_id += f"_frame{sample['idx']}"
        
        # 转换为 torch tensor
        csi_tensor = torch.from_numpy(csi_processed).float()
        skeleton_tensor = torch.from_numpy(skeleton_raw.astype(np.float32)).float()
        
        return {
            "csi": csi_tensor,
            "skeleton": skeleton_tensor,
            "label": label,
            "sample_id": sample_id
        }
    
    def get_class_distribution(self):
        """获取类别分布"""
        labels = []
        for i in range(len(self)):
            sample = self.official_dataset.data_list[i]
            action = sample['action']
            labels.append(self.action_to_label[action])
        
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))


def make_dataloader(dataset, batch_size, shuffle=True, num_workers=0):
    """
    创建数据加载器
    Args:
        dataset: MMFiDatasetWrapper 实例
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=shuffle,  # 训练时丢弃最后不完整的 batch
        pin_memory=True
    )


def create_dataloaders(config):
    """
    根据配置创建训练和验证数据加载器
    
    Args:
        config: 配置字典
    Returns:
        train_loader, val_loader, num_classes
    """
    # 解析划分比例
    split_str = config['dataset']['split']
    train_ratio = float(split_str.split(':')[0]) / 10
    test_ratio = float(split_str.split(':')[1]) / 10
    
    logger.info(f"数据集划分比例: {train_ratio:.0%} 训练 / {test_ratio:.0%} 测试")
    
    # 训练集
    train_dataset = MMFiDatasetWrapper(
        data_root=config['dataset']['root'],
        split_mode='train',
        split_ratio=train_ratio,
        protocol=config['dataset'].get('protocol', 'protocol2'),
        data_unit=config['dataset'].get('data_unit', 'sequence'),
        smooth_kernel_size=config['dataset'].get('smooth_kernel_size', 3),
        augmentation_config=config.get('augmentation', None),
        random_seed=config.get('random_seed', 0)
    )
    
    # 验证/测试集
    val_dataset = MMFiDatasetWrapper(
        data_root=config['dataset']['root'],
        split_mode='val',
        split_ratio=train_ratio,
        protocol=config['dataset'].get('protocol', 'protocol2'),
        data_unit=config['dataset'].get('data_unit', 'sequence'),
        smooth_kernel_size=config['dataset'].get('smooth_kernel_size', 3),
        augmentation_config=None,  # 验证集不增强
        random_seed=config.get('random_seed', 0)
    )
    
    # 验证划分比例
    total_samples = len(train_dataset) + len(val_dataset)
    actual_train_ratio = len(train_dataset) / total_samples
    actual_test_ratio = len(val_dataset) / total_samples
    
    logger.info(f"实际划分结果:")
    logger.info(f"  训练集: {len(train_dataset)} 样本 ({actual_train_ratio:.1%})")
    logger.info(f"  测试集: {len(val_dataset)} 样本 ({actual_test_ratio:.1%})")
    
    # 验证比例是否正确
    expected_ratio = train_ratio
    if abs(actual_train_ratio - expected_ratio) > 0.05:
        logger.warning(f"警告: 实际划分比例 ({actual_train_ratio:.1%}) 与期望 ({expected_ratio:.0%}) 偏差较大")
    
    # 创建 DataLoader
    train_loader = make_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 0)
    )
    
    val_loader = make_dataloader(
        val_dataset,
        batch_size=config['training'].get('val_batch_size', config['training']['batch_size']),
        shuffle=False,
        num_workers=config.get('num_workers', 0)
    )
    
    num_classes = len(train_dataset.action_to_label)
    
    return train_loader, val_loader, num_classes


def verify_data_split(config):
    """
    验证数据划分是否正确
    
    Args:
        config: 配置字典
    """
    logger.info("验证数据划分...")
    
    train_loader, val_loader, num_classes = create_dataloaders(config)
    
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    total_size = train_size + val_size
    
    print(f"\n{'='*50}")
    print(f"数据划分验证结果")
    print(f"{'='*50}")
    print(f"训练集大小: {train_size}")
    print(f"测试集大小: {val_size}")
    print(f"总样本数: {total_size}")
    print(f"训练集比例: {train_size/total_size:.1%}")
    print(f"测试集比例: {val_size/total_size:.1%}")
    print(f"类别数量: {num_classes}")
    print(f"{'='*50}\n")
    
    return train_size, val_size, num_classes
