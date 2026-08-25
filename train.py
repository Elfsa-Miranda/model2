#!/usr/bin/env python3
"""
主训练脚本

跨模态动量对比学习训练
- 训练时使用 CSI + RGB（骨架）
- 推理时仅使用 CSI

使用方法:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --resume checkpoints/best.pth
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import random

from datasets.mmfi_dataset import create_dataloaders
from models.csi_encoder import CSIEncoder
from models.rgb_encoder import RGBEncoder
from models.heads import HeadsModule
from trainer.trainer import Trainer


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(
        description='跨模态动量对比学习训练 - CSI-only HAR'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/default.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--resume', 
        type=str, 
        default=None,
        help='恢复训练的检查点路径'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='训练设备'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    print(f"加载配置: {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子
    set_seed(config.get('random_seed', 0))
    
    # 设置设备
    device = args.device
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, val_loader, num_classes = create_dataloaders(config)
    print(f"  训练集大小: {len(train_loader.dataset)}")
    print(f"  验证集大小: {len(val_loader.dataset)}")
    print(f"  类别数量: {num_classes}")
    
    # 创建 CSI 编码器
    print("创建 CSI 编码器...")
    csi_encoder = CSIEncoder(
        num_subcarriers=config['dataset']['num_subcarriers'],
        num_antennas=config['dataset']['num_antennas'],
        channel_dim=config['model']['csi']['channel_dim'],
        temporal_dim=config['model']['csi']['temporal_dim'],
        fusion_dim=config['model']['csi']['fusion_dim']
    )
    print(f"  CSI Encoder 参数量: {sum(p.numel() for p in csi_encoder.parameters())}")
    
    # 创建 RGB 编码器（参数冻结）
    print("创建 RGB 编码器...")
    rgb_encoder = RGBEncoder(
        num_joints=config['dataset']['num_joints'],
        coord_dim=3,
        spatial_dim=config['model']['rgb']['spatial_dim'],
        temporal_dim=config['model']['rgb']['temporal_dim'],
        fusion_dim=config['model']['rgb']['fusion_dim']
    )
    # 验证参数已冻结
    assert all(not p.requires_grad for p in rgb_encoder.parameters()), \
        "RGB Encoder 参数必须被冻结"
    print(f"  RGB Encoder 参数量: {sum(p.numel() for p in rgb_encoder.parameters())}")
    print(f"  RGB Encoder 参数已冻结: True")
    
    # 创建头部模块
    print("创建头部模块...")
    heads = HeadsModule(
        fusion_dim=config['model']['csi']['fusion_dim'],
        projector_dim=config['model']['projector_dim'],
        classifier_hidden=config['model']['classifier_hidden'],
        regressor_hidden=config['model']['regressor_hidden'],
        num_classes=num_classes,
        num_joints=config['dataset']['num_joints']
    )
    print(f"  Heads 可训练参数量: {sum(p.numel() for p in heads.get_trainable_params())}")
    
    # 创建训练器
    print("创建训练器...")
    trainer = Trainer(
        cfg=config,
        csi_encoder=csi_encoder,
        rgb_encoder=rgb_encoder,
        heads=heads,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    # 恢复训练
    if args.resume:
        print(f"恢复训练: {args.resume}")
        trainer.load_checkpoint(args.resume)
        print(f"  从 epoch {trainer.current_epoch} 继续训练")
    
    # 开始训练
    print("\n开始训练...")
    print("=" * 50)
    trainer.train()
    
    # 最终评估
    print("\n最终评估...")
    print("=" * 50)
    
    # 加载最佳模型
    best_checkpoint = os.path.join(config['output_folder'], 'best.pth')
    if os.path.exists(best_checkpoint):
        trainer.load_checkpoint(best_checkpoint)
    
    metrics = trainer.evaluate()
    print(f"测试准确率: {metrics['test_acc']:.2f}%")
    
    print("\n训练完成!")


if __name__ == '__main__':
    main()
