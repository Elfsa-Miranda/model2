"""
队列测试：验证内存队列的 FIFO 行为和梯度分离
"""
import unittest
import sys
import os
import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.queue import MemoryQueue


class TestMemoryQueue(unittest.TestCase):
    """测试内存队列"""
    
    def setUp(self):
        """测试初始化"""
        self.feature_dim = 128
        self.queue_size = 10
        self.queue = MemoryQueue(self.feature_dim, self.queue_size)
    
    def test_initial_state(self):
        """测试初始状态"""
        self.assertTrue(self.queue.is_empty())
        self.assertEqual(self.queue.size(), 0)
    
    def test_enqueue_basic(self):
        """测试基本入队操作"""
        keys = torch.randn(3, self.feature_dim)
        labels = torch.tensor([0, 1, 2])
        
        self.queue.enqueue(keys.detach(), labels)
        
        self.assertFalse(self.queue.is_empty())
        self.assertEqual(self.queue.size(), 3)
    
    def test_fifo_behavior(self):
        """测试队列的先进先出行为"""
        # 插入第一批
        keys1 = torch.ones(3, self.feature_dim) * 1
        labels1 = torch.tensor([0, 0, 0])
        self.queue.enqueue(keys1.detach(), labels1)
        
        # 插入第二批
        keys2 = torch.ones(3, self.feature_dim) * 2
        labels2 = torch.tensor([1, 1, 1])
        self.queue.enqueue(keys2.detach(), labels2)
        
        # 获取所有数据
        all_keys, all_labels = self.queue.get_all()
        
        # 验证顺序
        self.assertTrue(torch.allclose(all_keys[:3], keys1))
        self.assertTrue(torch.allclose(all_keys[3:6], keys2))
    
    def test_stored_tensors_no_grad(self):
        """测试存储的张量 requires_grad=False"""
        # 创建需要梯度的张量
        keys = torch.randn(3, self.feature_dim, requires_grad=True)
        
        # 入队时必须 detach
        self.queue.enqueue(keys.detach(), torch.tensor([0, 1, 2]))
        
        # 获取队列中的数据
        queue_keys = self.queue.get_queue()
        
        # 验证不需要梯度
        self.assertFalse(queue_keys.requires_grad)
    
    def test_enqueue_requires_detach(self):
        """测试入队时必须 detach"""
        keys = torch.randn(3, self.feature_dim, requires_grad=True)
        
        # 不 detach 应该抛出异常
        with self.assertRaises(AssertionError):
            self.queue.enqueue(keys, torch.tensor([0, 1, 2]))
    
    def test_queue_size_limit(self):
        """测试队列大小限制"""
        # 插入超过队列大小的数据
        for i in range(5):
            keys = torch.randn(3, self.feature_dim)
            self.queue.enqueue(keys.detach(), torch.tensor([i, i, i]))
        
        # 队列大小不应超过限制
        self.assertLessEqual(self.queue.size(), self.queue_size)
    
    def test_circular_buffer(self):
        """测试循环缓冲区行为"""
        # 填满队列
        for i in range(4):
            keys = torch.ones(3, self.feature_dim) * i
            self.queue.enqueue(keys.detach(), torch.tensor([i, i, i]))
        
        # 队列应该已满
        self.assertTrue(self.queue.is_full)
        
        # 继续插入，旧数据应该被覆盖
        new_keys = torch.ones(3, self.feature_dim) * 100
        self.queue.enqueue(new_keys.detach(), torch.tensor([100, 100, 100]))
        
        # 验证新数据在队列中
        all_keys, _ = self.queue.get_all()
        self.assertTrue(torch.any(all_keys == 100))
    
    def test_dequeue_operation(self):
        """测试出队操作"""
        # 插入数据
        keys = torch.randn(5, self.feature_dim)
        labels = torch.arange(5)
        self.queue.enqueue(keys.detach(), labels)
        
        # 出队
        dequeued_keys, dequeued_labels = self.queue.dequeue(3)
        
        self.assertEqual(dequeued_keys.shape[0], 3)
        self.assertEqual(dequeued_labels.shape[0], 3)
    
    def test_state_dict(self):
        """测试状态保存和加载"""
        # 插入数据
        keys = torch.randn(5, self.feature_dim)
        labels = torch.arange(5)
        self.queue.enqueue(keys.detach(), labels)
        
        # 保存状态
        state = self.queue.state_dict()
        
        # 创建新队列并加载状态
        new_queue = MemoryQueue(self.feature_dim, self.queue_size)
        new_queue.load_state_dict(state)
        
        # 验证状态一致
        self.assertEqual(new_queue.size(), self.queue.size())
        self.assertEqual(new_queue.ptr, self.queue.ptr)
    
    def test_reset(self):
        """测试重置操作"""
        # 插入数据
        keys = torch.randn(5, self.feature_dim)
        self.queue.enqueue(keys.detach(), torch.arange(5))
        
        # 重置
        self.queue.reset()
        
        # 验证已重置
        self.assertTrue(self.queue.is_empty())
        self.assertEqual(self.queue.size(), 0)


if __name__ == '__main__':
    unittest.main()
