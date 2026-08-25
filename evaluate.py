#!/usr/bin/env python3
"""
评估脚本

【关键约束】评估时只使用 CSI Encoder + Classifier
- 不使用 RGB 数据
- 不使用内存队列
- 不使用回归头
- 不使用投影头

使用方法:
    python evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pth
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from datasets.mmfi_dataset import create_dataloaders
from models.csi_encoder import CSIEncoder
from models.heads import ClassifierHead


def load_model_for_inference(config, checkpoint_path, device):
    """
    加载推理模型
    
    【关键】只加载 CSI Encoder + Classifier
    不加载 RGB Encoder、Projector、Regressor、Queue
    """
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 创建 CSI 编码器
    csi_encoder = CSIEncoder(
        num_subcarriers=config['dataset']['num_subcarriers'],
        num_antennas=config['dataset']['num_antennas'],
        channel_dim=config['model']['csi']['channel_dim'],
        temporal_dim=config['model']['csi']['temporal_dim'],
        fusion_dim=config['model']['csi']['fusion_dim']
    ).to(device)
    
    # 加载 CSI 编码器权重
    csi_encoder.load_state_dict(checkpoint['csi_encoder'])
    csi_encoder.eval()
    
    # 创建分类器
    # 从 heads 中提取分类器权重
    heads_state = checkpoint['heads']
    classifier_state = {
        k.replace('classifier.', ''): v 
        for k, v in heads_state.items() 
        if k.startswith('classifier.')
    }
    
    # 获取类别数量
    num_classes = classifier_state['classifier.4.weight'].shape[0]
    
    classifier = ClassifierHead(
        input_dim=config['model']['csi']['fusion_dim'],
        hidden_dim=config['model']['classifier_hidden'],
        num_classes=num_classes
    ).to(device)
    
    classifier.load_state_dict(classifier_state)
    classifier.eval()
    
    return csi_encoder, classifier


@torch.no_grad()
def evaluate(csi_encoder, classifier, test_loader, device):
    """
    评估模型
    
    【关键约束】
    - 只使用 CSI Encoder + Classifier
    - 不使用 RGB、Queue、Regressor、Projector
    """
    csi_encoder.eval()
    classifier.eval()
    
    all_preds = []
    all_labels = []
    
    for batch in test_loader:
        csi = batch['csi'].to(device)
        labels = batch['label']
        
        # 【推理】只使用 CSI Encoder + Classifier
        f_csi = csi_encoder(csi)
        logits = classifier(f_csi)
        
        _, predicted = logits.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)


def compute_metrics(preds, labels, class_names=None):
    """计算评估指标"""
    # 准确率
    accuracy = 100. * np.mean(preds == labels)
    
    # F1 分数
    f1_macro = f1_score(labels, preds, average='macro') * 100
    f1_weighted = f1_score(labels, preds, average='weighted') * 100
    
    # 混淆矩阵
    cm = confusion_matrix(labels, preds)
    
    # 每类准确率
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    
    # 分类报告
    report = classification_report(labels, preds, target_names=class_names)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'per_class_accuracy': per_class_acc,
        'classification_report': report
    }


def plot_confusion_matrix(cm, class_names, save_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='CSI-only HAR 评估'
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
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='评估设备'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./evaluation_results',
        help='结果保存目录'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    print(f"加载配置: {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = args.device
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    print("创建数据加载器...")
    _, val_loader, num_classes = create_dataloaders(config)
    print(f"  验证集大小: {len(val_loader.dataset)}")
    
    # 加载模型（只加载 CSI Encoder + Classifier）
    print(f"加载模型: {args.checkpoint}")
    csi_encoder, classifier = load_model_for_inference(
        config, args.checkpoint, device
    )
    print("  【推理模式】只使用 CSI Encoder + Classifier")
    print("  【禁止项】不使用 RGB、Queue、Regressor、Projector")
    
    # 评估
    print("\n开始评估...")
    preds, labels = evaluate(csi_encoder, classifier, val_loader, device)
    
    # 计算指标
    # 获取类别名称
    if config['dataset']['protocol'] == 'protocol1':
        class_names = ['A02', 'A03', 'A04', 'A05', 'A13', 'A14', 'A17', 'A18', 
                      'A19', 'A20', 'A21', 'A22', 'A23', 'A27']
    elif config['dataset']['protocol'] == 'protocol2':
        class_names = ['A01', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 
                      'A15', 'A16', 'A24', 'A25', 'A26']
    else:
        class_names = [f'A{i:02d}' for i in range(1, num_classes + 1)]
    
    metrics = compute_metrics(preds, labels, class_names)
    
    # 打印结果
    print("\n" + "=" * 50)
    print("评估结果")
    print("=" * 50)
    print(f"准确率: {metrics['accuracy']:.2f}%")
    print(f"F1 (Macro): {metrics['f1_macro']:.2f}%")
    print(f"F1 (Weighted): {metrics['f1_weighted']:.2f}%")
    print("\n每类准确率:")
    for i, (name, acc) in enumerate(zip(class_names, metrics['per_class_accuracy'])):
        print(f"  {name}: {acc:.2f}%")
    print("\n分类报告:")
    print(metrics['classification_report'])
    
    # 保存结果
    os.makedirs(args.output, exist_ok=True)
    
    # 保存混淆矩阵图
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names,
        os.path.join(args.output, 'confusion_matrix.png')
    )
    
    # 保存指标
    results = {
        'accuracy': float(metrics['accuracy']),
        'f1_macro': float(metrics['f1_macro']),
        'f1_weighted': float(metrics['f1_weighted']),
        'per_class_accuracy': {
            name: float(acc) 
            for name, acc in zip(class_names, metrics['per_class_accuracy'])
        }
    }
    
    import json
    with open(os.path.join(args.output, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n指标已保存: {os.path.join(args.output, 'metrics.json')}")
    
    print("\n评估完成!")


if __name__ == '__main__':
    main()
