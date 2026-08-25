"""
Memory Queue for Contrastive Learning

核心功能：
- 存储历史 RGB key embeddings 作为负样本
- FIFO（先进先出）队列行为
- 存储的张量不需要梯度

关键约束：
- 只存储 RGB keys，不存储 CSI keys
- 存储的张量必须 detach()（无梯度）
- 支持保存和加载状态
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class MemoryQueue:
    """
    内存队列：存储 RGB key embeddings
    
    特性：
    - FIFO 队列行为
    - 固定大小，自动淘汰旧数据
    - 存储的张量无梯度
    - 可选存储标签（用于 Hard Negative）
    """
    
    def __init__(self, 
                 feature_dim: int,
                 queue_size: int = 4096,
                 device: str = 'cpu'):
        """
        Args:
            feature_dim: 特征维度
            queue_size: 队列大小
            device: 设备
        """
        self.feature_dim = feature_dim
        self.queue_size = queue_size
        self.device = device
        
        # 初始化队列（使用 register_buffer 风格，但不是 nn.Module）
        self.queue = torch.zeros(queue_size, feature_dim, device=device)
        self.queue_labels = torch.zeros(queue_size, dtype=torch.long, device=device)
        
        # 队列指针
        self.ptr = 0
        self.is_full = False
    
    @torch.no_grad()
    def enqueue(self, keys: torch.Tensor, labels: Optional[torch.Tensor] = None) -> None:
        """
        入队操作
        
        Args:
            keys: RGB key embeddings [B, D]（必须已 detach）
            labels: 对应的标签 [B]（可选）
        
        【关键约束】
        - keys 必须是 detach() 后的张量
        - 只存储 RGB keys，不存储 CSI keys
        """
        # 验证输入
        assert not keys.requires_grad, "入队的张量必须已 detach（requires_grad=False）"
        
        batch_size = keys.shape[0]
        
        # 确保在正确的设备上
        keys = keys.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        
        # 计算可用空间
        if self.ptr + batch_size <= self.queue_size:
            # 直接插入
            self.queue[self.ptr:self.ptr + batch_size] = keys
            if labels is not None:
                self.queue_labels[self.ptr:self.ptr + batch_size] = labels
            self.ptr += batch_size
        else:
            # 需要循环插入
            remaining = self.queue_size - self.ptr
            self.queue[self.ptr:] = keys[:remaining]
            self.queue[:batch_size - remaining] = keys[remaining:]
            if labels is not None:
                self.queue_labels[self.ptr:] = labels[:remaining]
                self.queue_labels[:batch_size - remaining] = labels[remaining:]
            self.ptr = batch_size - remaining
            self.is_full = True
        
        # 检查是否已满
        if self.ptr >= self.queue_size:
            self.ptr = 0
            self.is_full = True
    
    def dequeue(self, num: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        出队操作（获取最旧的数据）
        
        Args:
            num: 出队数量
        
        Returns:
            keys: [num, D]
            labels: [num]（如果有）
        """
        if self.is_full:
            # 队列已满，从 ptr 位置开始是最旧的数据
            start = self.ptr
            if start + num <= self.queue_size:
                keys = self.queue[start:start + num].clone()
                labels = self.queue_labels[start:start + num].clone()
            else:
                remaining = self.queue_size - start
                keys = torch.cat([
                    self.queue[start:],
                    self.queue[:num - remaining]
                ], dim=0)
                labels = torch.cat([
                    self.queue_labels[start:],
                    self.queue_labels[:num - remaining]
                ], dim=0)
        else:
            # 队列未满，从头开始
            keys = self.queue[:min(num, self.ptr)].clone()
            labels = self.queue_labels[:min(num, self.ptr)].clone()
        
        return keys, labels
    
    def get_all(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取队列中所有数据
        
        Returns:
            keys: [N, D]（N 为当前队列中的数据量）
            labels: [N]
        """
        if self.is_full:
            return self.queue.clone(), self.queue_labels.clone()
        else:
            return self.queue[:self.ptr].clone(), self.queue_labels[:self.ptr].clone()
    
    def get_queue(self) -> torch.Tensor:
        """
        获取队列中的 keys（用于对比学习）
        
        Returns:
            keys: [N, D]
        """
        if self.is_full:
            return self.queue
        else:
            return self.queue[:self.ptr]
    
    def get_labels(self) -> torch.Tensor:
        """
        获取队列中的标签
        
        Returns:
            labels: [N]
        """
        if self.is_full:
            return self.queue_labels
        else:
            return self.queue_labels[:self.ptr]
    
    def size(self) -> int:
        """返回当前队列中的数据量"""
        if self.is_full:
            return self.queue_size
        else:
            return self.ptr
    
    def is_empty(self) -> bool:
        """检查队列是否为空"""
        return self.ptr == 0 and not self.is_full
    
    def to(self, device: str) -> 'MemoryQueue':
        """移动到指定设备"""
        self.device = device
        self.queue = self.queue.to(device)
        self.queue_labels = self.queue_labels.to(device)
        return self
    
    def state_dict(self) -> dict:
        """保存状态"""
        return {
            'queue': self.queue.cpu(),
            'queue_labels': self.queue_labels.cpu(),
            'ptr': self.ptr,
            'is_full': self.is_full,
            'feature_dim': self.feature_dim,
            'queue_size': self.queue_size
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """加载状态"""
        self.queue = state_dict['queue'].to(self.device)
        self.queue_labels = state_dict['queue_labels'].to(self.device)
        self.ptr = state_dict['ptr']
        self.is_full = state_dict['is_full']
    
    def reset(self) -> None:
        """重置队列"""
        self.queue.zero_()
        self.queue_labels.zero_()
        self.ptr = 0
        self.is_full = False


# 测试代码
if __name__ == '__main__':
    print("测试 MemoryQueue...")
    
    feature_dim = 128
    queue_size = 10
    batch_size = 3
    
    # 创建队列
    queue = MemoryQueue(feature_dim, queue_size)
    print(f"初始队列大小: {queue.size()}")
    print(f"队列是否为空: {queue.is_empty()}")
    
    # 测试入队
    print("\n测试入队...")
    for i in range(5):
        keys = torch.randn(batch_size, feature_dim)
        labels = torch.tensor([i * 3, i * 3 + 1, i * 3 + 2])
        queue.enqueue(keys.detach(), labels)
        print(f"  入队 batch {i+1}, 当前队列大小: {queue.size()}, is_full: {queue.is_full}")
    
    # 测试获取所有数据
    print("\n测试获取所有数据...")
    all_keys, all_labels = queue.get_all()
    print(f"  所有 keys 形状: {all_keys.shape}")
    print(f"  所有 labels: {all_labels}")
    
    # 测试 FIFO 行为
    print("\n测试 FIFO 行为...")
    old_keys, old_labels = queue.dequeue(3)
    print(f"  出队 keys 形状: {old_keys.shape}")
    print(f"  出队 labels: {old_labels}")
    
    # 测试 requires_grad 检查
    print("\n测试 requires_grad 检查...")
    try:
        bad_keys = torch.randn(batch_size, feature_dim, requires_grad=True)
        queue.enqueue(bad_keys)
        print("  错误：应该抛出异常")
    except AssertionError as e:
        print(f"  正确：捕获到异常 - {e}")
    
    # 测试状态保存和加载
    print("\n测试状态保存和加载...")
    state = queue.state_dict()
    new_queue = MemoryQueue(feature_dim, queue_size)
    new_queue.load_state_dict(state)
    print(f"  加载后队列大小: {new_queue.size()}")
    print(f"  加载后 is_full: {new_queue.is_full}")
