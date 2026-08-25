"""
Supervised Contrastive Loss (SupCon)

基于 Khosla et al. "Supervised Contrastive Learning" 的实现

核心思想：
- 正样本：同一实例的 CSI-RGB 对
- 负样本：来自内存队列的 RGB keys + batch 内其他样本
- 使用标签信息增强对比学习

关键特性：
- 支持跨模态对比（CSI query vs RGB key）
- 支持内存队列提供大量负样本
- 支持 Hard Negative 采样（基于标签）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    监督对比损失
    
    公式：
    L = -1/|P(i)| * Σ_{p∈P(i)} log(exp(z_i·z_p/τ) / Σ_{a∈A(i)} exp(z_i·z_a/τ))
    
    其中：
    - P(i): 正样本集合（同一实例的 CSI-RGB 对）
    - A(i): 所有样本集合（正样本 + 负样本）
    - τ: 温度参数
    """
    
    def __init__(self, temperature=0.07):
        """
        Args:
            temperature: 温度参数，控制分布的平滑程度
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_csi, z_rgb, labels=None, queue=None, queue_labels=None):
        """
        计算监督对比损失
        
        Args:
            z_csi: CSI 特征 [B, D]（query，L2 归一化）
            z_rgb: RGB 特征 [B, D]（key，L2 归一化）
            labels: 标签 [B]（可选，用于监督对比）
            queue: 内存队列 [N, D]（可选，提供额外负样本）
            queue_labels: 队列标签 [N]（可选，用于 Hard Negative）
        
        Returns:
            loss: 对比损失标量
        """
        batch_size = z_csi.shape[0]
        device = z_csi.device
        
        # 1. 计算正样本相似度（同一实例的 CSI-RGB 对）
        # pos_sim[i] = z_csi[i] · z_rgb[i] / τ
        pos_sim = torch.sum(z_csi * z_rgb, dim=1) / self.temperature  # [B]
        
        # 2. 计算负样本相似度
        # 负样本来源：
        # a) batch 内其他样本的 RGB
        # b) 内存队列中的 RGB keys
        
        # 2a. Batch 内负样本
        # neg_sim_batch[i, j] = z_csi[i] · z_rgb[j] / τ (i ≠ j)
        neg_sim_batch = torch.mm(z_csi, z_rgb.t()) / self.temperature  # [B, B]
        
        # 2b. 队列负样本
        if queue is not None and queue.shape[0] > 0:
            # neg_sim_queue[i, k] = z_csi[i] · queue[k] / τ
            neg_sim_queue = torch.mm(z_csi, queue.t()) / self.temperature  # [B, N]
            # 合并所有负样本
            all_sim = torch.cat([neg_sim_batch, neg_sim_queue], dim=1)  # [B, B+N]
        else:
            all_sim = neg_sim_batch  # [B, B]
        
        # 3. 构建 mask（排除自身作为负样本）
        # 对于 batch 内，对角线是正样本，需要特殊处理
        
        # 4. 计算 InfoNCE 损失
        # L = -log(exp(pos_sim) / (exp(pos_sim) + Σ exp(neg_sim)))
        
        # 方法：将正样本放在第一列，然后计算 softmax
        # logits[i, 0] = pos_sim[i]
        # logits[i, 1:] = neg_sim[i, :]（排除自身）
        
        # 创建 logits 矩阵
        # 正样本在第一列
        pos_sim = pos_sim.unsqueeze(1)  # [B, 1]
        
        # 负样本：排除对角线（自身）
        mask = torch.eye(batch_size, device=device).bool()
        neg_sim_batch_masked = neg_sim_batch.masked_fill(mask, float('-inf'))
        
        if queue is not None and queue.shape[0] > 0:
            logits = torch.cat([pos_sim, neg_sim_batch_masked, neg_sim_queue], dim=1)  # [B, 1+B+N]
        else:
            logits = torch.cat([pos_sim, neg_sim_batch_masked], dim=1)  # [B, 1+B]
        
        # 标签：正样本在第 0 位
        targets = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(logits, targets)
        
        return loss


class CrossModalSupConLoss(nn.Module):
    """
    跨模态监督对比损失（增强版）
    
    特性：
    - 支持标签感知的正负样本定义
    - 支持 Hard Negative 采样
    - 支持内存队列
    """
    
    def __init__(self, temperature=0.07, use_hard_negative=True):
        """
        Args:
            temperature: 温度参数
            use_hard_negative: 是否使用 Hard Negative（同类别但不同实例）
        """
        super().__init__()
        self.temperature = temperature
        self.use_hard_negative = use_hard_negative
    
    def forward(self, z_csi, z_rgb, labels, queue=None, queue_labels=None):
        """
        计算跨模态监督对比损失
        
        Args:
            z_csi: CSI 特征 [B, D]（query，L2 归一化）
            z_rgb: RGB 特征 [B, D]（key，L2 归一化）
            labels: 标签 [B]
            queue: 内存队列 [N, D]（可选）
            queue_labels: 队列标签 [N]（可选）
        
        Returns:
            loss: 对比损失标量
        """
        batch_size = z_csi.shape[0]
        device = z_csi.device
        
        # 1. 计算所有相似度
        # 正样本：同一实例的 CSI-RGB 对
        pos_sim = torch.sum(z_csi * z_rgb, dim=1, keepdim=True) / self.temperature  # [B, 1]
        
        # Batch 内相似度
        sim_batch = torch.mm(z_csi, z_rgb.t()) / self.temperature  # [B, B]
        
        # 队列相似度
        if queue is not None and queue.shape[0] > 0:
            sim_queue = torch.mm(z_csi, queue.t()) / self.temperature  # [B, N]
        else:
            sim_queue = None
        
        # 2. 构建正负样本 mask
        # 正样本 mask：同一实例（对角线）
        pos_mask = torch.eye(batch_size, device=device).bool()
        
        # 负样本 mask：不同实例
        neg_mask_batch = ~pos_mask
        
        # 3. 如果使用 Hard Negative，调整 mask
        if self.use_hard_negative and labels is not None:
            # 同类别的样本作为 Hard Negative（更难区分）
            label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
            # Hard Negative: 同类别但不同实例
            hard_neg_mask = label_matrix & neg_mask_batch
            # 普通负样本权重降低（可选）
        
        # 4. 计算损失
        # 使用 InfoNCE 风格的损失
        
        # 构建 logits
        # 排除对角线（自身）
        sim_batch_masked = sim_batch.masked_fill(pos_mask, float('-inf'))
        
        if sim_queue is not None:
            all_neg_sim = torch.cat([sim_batch_masked, sim_queue], dim=1)  # [B, B+N]
        else:
            all_neg_sim = sim_batch_masked  # [B, B]
        
        # logits: [正样本, 负样本]
        logits = torch.cat([pos_sim, all_neg_sim], dim=1)  # [B, 1+B+N] 或 [B, 1+B]
        
        # 标签：正样本在第 0 位
        targets = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(logits, targets)
        
        return loss


# 测试代码
if __name__ == '__main__':
    batch_size = 8
    feature_dim = 128
    queue_size = 64
    
    # 创建测试数据
    z_csi = F.normalize(torch.randn(batch_size, feature_dim), dim=1)
    z_rgb = F.normalize(torch.randn(batch_size, feature_dim), dim=1)
    labels = torch.randint(0, 10, (batch_size,))
    queue = F.normalize(torch.randn(queue_size, feature_dim), dim=1)
    queue_labels = torch.randint(0, 10, (queue_size,))
    
    # 测试 SupConLoss
    print("测试 SupConLoss...")
    criterion = SupConLoss(temperature=0.07)
    loss = criterion(z_csi, z_rgb, labels, queue, queue_labels)
    print(f"  Loss: {loss.item():.4f}")
    
    # 测试 CrossModalSupConLoss
    print("\n测试 CrossModalSupConLoss...")
    criterion2 = CrossModalSupConLoss(temperature=0.07, use_hard_negative=True)
    loss2 = criterion2(z_csi, z_rgb, labels, queue, queue_labels)
    print(f"  Loss: {loss2.item():.4f}")
    
    # 测试无队列情况
    print("\n测试无队列情况...")
    loss3 = criterion(z_csi, z_rgb)
    print(f"  Loss (无队列): {loss3.item():.4f}")
