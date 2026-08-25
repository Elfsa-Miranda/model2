#!/usr/bin/env python3
"""
推理脚本 - 仅使用 CSI 数据

【关键约束】推理阶段严格禁止使用：
- RGB 数据
- 内存队列
- 回归头
- 投影头

只使用：
- CSI Encoder
- Classifier Head

使用方法:
    python inference.py --config configs/default.yaml --checkpoint checkpoints/best.pth --input path/to/csi_data
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np

from models.csi_encoder import CSIEncoder
from models.heads import ClassifierHead


class CSIInferenceModel:
    """
    CSI 推理模型
    
    【关键约束】
    - 只包含 CSI Encoder + Classifier
    - 不包含 RGB Encoder、Projector、Regressor、Queue
    """
    
    def __init__(self, config, checkpoint_path, device='cpu'):
        """
        Args:
            config: 配置字典
            checkpoint_path: 检查点路径
            device: 推理设备
        """
        self.config = config
        self.device = device
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 【推理】只创建 CSI Encoder
        self.csi_encoder = CSIEncoder(
            num_subcarriers=config['dataset']['num_subcarriers'],
            num_antennas=config['dataset']['num_antennas'],
            channel_dim=config['model']['csi']['channel_dim'],
            temporal_dim=config['model']['csi']['temporal_dim'],
            fusion_dim=config['model']['csi']['fusion_dim']
        ).to(device)
        
        # 加载 CSI Encoder 权重
        self.csi_encoder.load_state_dict(checkpoint['csi_encoder'])
        self.csi_encoder.eval()
        
        # 【推理】只创建 Classifier
        heads_state = checkpoint['heads']
        classifier_state = {
            k.replace('classifier.', ''): v 
            for k, v in heads_state.items() 
            if k.startswith('classifier.')
        }
        
        num_classes = classifier_state['classifier.4.weight'].shape[0]
        
        self.classifier = ClassifierHead(
            input_dim=config['model']['csi']['fusion_dim'],
            hidden_dim=config['model']['classifier_hidden'],
            num_classes=num_classes
        ).to(device)
        
        self.classifier.load_state_dict(classifier_state)
        self.classifier.eval()
        
        # 类别映射
        self._build_class_mapping()
        
        # 【验证】确保没有加载禁止的组件
        self._verify_inference_constraints()
    
    def _build_class_mapping(self):
        """构建类别映射"""
        protocol = self.config['dataset']['protocol']
        if protocol == 'protocol1':
            actions = ['A02', 'A03', 'A04', 'A05', 'A13', 'A14', 'A17', 'A18', 
                      'A19', 'A20', 'A21', 'A22', 'A23', 'A27']
        elif protocol == 'protocol2':
            actions = ['A01', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 
                      'A15', 'A16', 'A24', 'A25', 'A26']
        else:
            actions = [f'A{i:02d}' for i in range(1, 28)]
        
        self.idx_to_action = {i: a for i, a in enumerate(sorted(actions))}
    
    def _verify_inference_constraints(self):
        """验证推理约束"""
        # 确保没有 RGB Encoder
        assert not hasattr(self, 'rgb_encoder'), \
            "【禁止项】推理时不能使用 RGB Encoder"
        
        # 确保没有 Projector
        assert not hasattr(self, 'projector'), \
            "【禁止项】推理时不能使用 Projector"
        
        # 确保没有 Regressor
        assert not hasattr(self, 'regressor'), \
            "【禁止项】推理时不能使用 Regressor"
        
        # 确保没有 Queue
        assert not hasattr(self, 'queue'), \
            "【禁止项】推理时不能使用 Queue"
    
    def preprocess_csi(self, csi_raw):
        """
        CSI 预处理
        
        Args:
            csi_raw: 原始 CSI 数据 [T, Subcarriers, Antennas]
        
        Returns:
            csi_processed: 预处理后的 CSI [1, T, Subcarriers, Antennas]
        """
        csi = csi_raw.copy()
        
        # Per-subcarrier 减均值
        csi_mean = np.mean(csi, axis=0, keepdims=True)
        csi = csi - csi_mean
        
        # 时间平滑
        kernel_size = self.config['dataset'].get('smooth_kernel_size', 3)
        if kernel_size > 1:
            kernel = np.ones(kernel_size) / kernel_size
            T, S, A = csi.shape
            csi_smoothed = np.zeros_like(csi)
            for s in range(S):
                for a in range(A):
                    csi_smoothed[:, s, a] = np.convolve(csi[:, s, a], kernel, mode='same')
            csi = csi_smoothed
        
        # 归一化
        csi_min = np.min(csi)
        csi_max = np.max(csi)
        if csi_max - csi_min > 1e-6:
            csi = (csi - csi_min) / (csi_max - csi_min)
        
        # 转换为 tensor 并添加 batch 维度
        csi_tensor = torch.from_numpy(csi).float().unsqueeze(0)
        
        return csi_tensor
    
    @torch.no_grad()
    def predict(self, csi):
        """
        预测活动类别
        
        【关键】只使用 CSI Encoder + Classifier
        
        Args:
            csi: CSI 数据 [B, T, Subcarriers, Antennas] 或 [T, Subcarriers, Antennas]
        
        Returns:
            predictions: 预测类别索引
            probabilities: 预测概率
            action_names: 预测动作名称
        """
        # 确保是 tensor
        if isinstance(csi, np.ndarray):
            csi = torch.from_numpy(csi).float()
        
        # 添加 batch 维度
        if csi.dim() == 3:
            csi = csi.unsqueeze(0)
        
        csi = csi.to(self.device)
        
        # 【推理】CSI Encoder -> Classifier
        f_csi = self.csi_encoder(csi)
        logits = self.classifier(f_csi)
        
        # 计算概率
        probs = torch.softmax(logits, dim=1)
        
        # 获取预测
        _, predicted = logits.max(1)
        
        predictions = predicted.cpu().numpy()
        probabilities = probs.cpu().numpy()
        action_names = [self.idx_to_action[p] for p in predictions]
        
        return predictions, probabilities, action_names
    
    def predict_single(self, csi_raw):
        """
        预测单个样本
        
        Args:
            csi_raw: 原始 CSI 数据 [T, Subcarriers, Antennas]
        
        Returns:
            action: 预测动作名称
            confidence: 置信度
        """
        csi = self.preprocess_csi(csi_raw)
        predictions, probabilities, action_names = self.predict(csi)
        
        action = action_names[0]
        confidence = probabilities[0].max()
        
        return action, confidence


def main():
    parser = argparse.ArgumentParser(
        description='CSI-only HAR 推理'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/default.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        required=True,
        help='模型检查点路径'
    )
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help='输入 CSI 数据路径（.npy 文件或目录）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='推理设备'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='结果保存路径'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    print(f"加载配置: {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建推理模型
    print(f"加载模型: {args.checkpoint}")
    model = CSIInferenceModel(config, args.checkpoint, args.device)
    print("  【推理模式】只使用 CSI Encoder + Classifier")
    print("  【禁止项】不使用 RGB、Queue、Regressor、Projector")
    
    # 加载输入数据
    print(f"\n加载输入数据: {args.input}")
    
    if os.path.isfile(args.input):
        # 单个文件
        csi_data = np.load(args.input)
        print(f"  数据形状: {csi_data.shape}")
        
        # 预测
        action, confidence = model.predict_single(csi_data)
        print(f"\n预测结果:")
        print(f"  动作: {action}")
        print(f"  置信度: {confidence:.4f}")
        
    elif os.path.isdir(args.input):
        # 目录（批量推理）
        import glob
        files = sorted(glob.glob(os.path.join(args.input, '*.npy')))
        print(f"  找到 {len(files)} 个文件")
        
        results = []
        for f in files:
            csi_data = np.load(f)
            action, confidence = model.predict_single(csi_data)
            results.append({
                'file': os.path.basename(f),
                'action': action,
                'confidence': float(confidence)
            })
            print(f"  {os.path.basename(f)}: {action} ({confidence:.4f})")
        
        # 保存结果
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n结果已保存: {args.output}")
    
    else:
        print(f"错误: 输入路径不存在: {args.input}")
        sys.exit(1)
    
    print("\n推理完成!")


if __name__ == '__main__':
    main()
