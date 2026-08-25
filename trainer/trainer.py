"""
Trainer for Cross-modal Momentum Contrastive Learning

训练流程（黄金骨架）：
1. CSI forward -> fused feature f_csi
2. RGB forward (torch.no_grad()) -> f_rgb
3. Project: z_csi = projector_csi(f_csi), z_rgb = projector_rgb(f_rgb)
4. Compute losses: L_con, L_cls, L_reg
5. loss.backward()  # only CSI graph
6. optimizer.step()
7. momentum_update(rgb_model, csi_model, m)
8. queue.enqueue(z_rgb.detach())  # store rgb keys

关键约束：
- RGB forward 必须在 torch.no_grad() 中
- optimizer 不包含 RGB 参数
- EMA 更新在 optimizer.step() 之后
- queue 只存储 RGB keys
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional, Tuple, List
import json
import logging
from datetime import datetime

from losses.supcon import SupConLoss, CrossModalSupConLoss
from utils.queue import MemoryQueue

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        self.metrics_file = os.path.join(log_dir, 'metrics.json')
        
        self.metrics_history = []
    
    def log(self, message: str):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        print(log_message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def log_metrics(self, metrics: Dict):
        """记录指标"""
        self.metrics_history.append(metrics)
        
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def get_metrics_history(self) -> List[Dict]:
        """获取指标历史"""
        return self.metrics_history


class Trainer:
    """
    跨模态动量对比学习训练器
    
    关键约束（必须严格遵守）：
    1. RGB forward 使用 torch.no_grad() 包裹
    2. optimizer 不包含 rgb_encoder 参数
    3. EMA 更新在 optimizer.step() 之后
    4. queue 只存储 RGB keys（detach）
    """
    
    def __init__(self,
                 cfg: dict,
                 csi_encoder: nn.Module,
                 rgb_encoder: nn.Module,
                 heads: nn.Module,
                 train_loader,
                 val_loader,
                 device: str = 'cuda'):
        """
        Args:
            cfg: 配置字典
            csi_encoder: CSI 编码器
            rgb_encoder: RGB 编码器（参数已冻结）
            heads: 头部模块（projector, classifier, regressor）
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            device: 设备
        """
        self.cfg = cfg
        self.device = device
        
        # 模型
        self.csi_encoder = csi_encoder.to(device)
        self.rgb_encoder = rgb_encoder.to(device)
        self.heads = heads.to(device)
        
        # 数据加载器
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 日志记录器
        log_dir = cfg.get('log_dir', './logs')
        self.logger = TrainingLogger(log_dir)
        
        # TensorBoard 记录器
        tensorboard_dir = os.path.join(log_dir, 'tensorboard')
        self.writer = SummaryWriter(tensorboard_dir)
        
        # 【关键验证】RGB Encoder 参数必须冻结
        self._verify_rgb_frozen()
        
        # 创建优化器（不包含 RGB 参数）
        self.optimizer = self._create_optimizer()
        
        # 【关键验证】优化器不包含 RGB 参数
        self._verify_optimizer_excludes_rgb()
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 损失函数
        self.contrastive_loss = CrossModalSupConLoss(
            temperature=float(cfg['training']['temperature'])
        )
        
        # 分类损失（支持标签平滑）
        label_smoothing = float(cfg['training'].get('label_smoothing', 0.0))
        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # 回归损失（支持多种类型）
        regression_loss_type = cfg['training'].get('regression_loss_type', 'smooth_l1')
        if regression_loss_type == 'mse':
            self.regression_loss = nn.MSELoss()
        elif regression_loss_type == 'l1':
            self.regression_loss = nn.L1Loss()
        elif regression_loss_type == 'smooth_l1':
            self.regression_loss = nn.SmoothL1Loss(beta=1.0)
        elif regression_loss_type == 'huber':
            self.regression_loss = nn.HuberLoss(delta=1.0)
        else:
            self.regression_loss = nn.SmoothL1Loss(beta=1.0)  # 默认使用 Smooth L1
        
        # 内存队列（只存储 RGB keys）
        self.queue = MemoryQueue(
            feature_dim=int(cfg['model']['projector_dim']),
            queue_size=int(cfg['training']['queue_size']),
            device=device
        )
        
        # 注意：由于 RGB Encoder 和 CSI Encoder 结构不同（输入维度不同），
        # 我们不对整个 encoder 进行 EMA 更新，只对 projector 进行 EMA 更新
        # 这符合跨模态对比学习的设计：不同模态可以有不同的编码器结构
        
        # 训练状态
        self.current_epoch = 0
        self.best_metric = 0.0
        self.training_history = []
        
        # 早停机制
        early_stop_cfg = cfg.get('training', {}).get('early_stopping', {})
        self.early_stopping_enabled = early_stop_cfg.get('enabled', False)
        self.early_stopping_patience = early_stop_cfg.get('patience', 20)
        self.early_stopping_min_delta = early_stop_cfg.get('min_delta', 0.001)
        self.early_stopping_monitor = early_stop_cfg.get('monitor', 'val_acc')
        self.early_stopping_counter = 0
        self.best_early_stop_metric = 0.0 if 'acc' in self.early_stopping_monitor else float('inf')
        
        # 记录初始化信息
        self._log_initialization()
    
    def _log_initialization(self):
        """记录初始化信息"""
        self.logger.log("=" * 60)
        self.logger.log("跨模态动量对比学习训练器初始化")
        self.logger.log("=" * 60)
        self.logger.log(f"设备: {self.device}")
        self.logger.log(f"训练集大小: {len(self.train_loader.dataset)}")
        self.logger.log(f"验证集大小: {len(self.val_loader.dataset)}")
        self.logger.log(f"批次大小: {self.cfg['training']['batch_size']}")
        self.logger.log(f"训练轮数: {self.cfg['training']['epochs']}")
        self.logger.log(f"学习率: {self.cfg['training']['lr']}")
        self.logger.log(f"EMA 动量: {self.cfg['training']['ema_m']}")
        self.logger.log(f"队列大小: {self.cfg['training']['queue_size']}")
        self.logger.log(f"温度参数: {self.cfg['training']['temperature']}")
        
        # Checkpoint 配置
        save_interval = self.cfg.get('checkpoint', {}).get('save_interval', 10)
        self.logger.log(f"Checkpoint 保存间隔: {save_interval if save_interval > 0 else '不定期保存'}")
        
        self.logger.log("=" * 60)
        
        # 验证约束
        self.logger.log("约束验证:")
        self.logger.log(f"  - RGB Encoder 参数冻结: ✓")
        self.logger.log(f"  - Optimizer 不包含 RGB 参数: ✓")
        self.logger.log(f"  - Queue 只存储 RGB keys: ✓")
    
    def _verify_rgb_frozen(self):
        """验证 RGB Encoder 参数已冻结"""
        assert all(not p.requires_grad for p in self.rgb_encoder.parameters()), \
            "【禁止项】RGB Encoder 参数必须被冻结（requires_grad=False）"
        
        # 同时验证 RGB Projector
        assert all(not p.requires_grad for p in self.heads.rgb_projector.parameters()), \
            "【禁止项】RGB Projector 参数必须被冻结"
    
    def _verify_optimizer_excludes_rgb(self):
        """验证优化器不包含 RGB 参数"""
        rgb_param_ids = set(id(p) for p in self.rgb_encoder.parameters())
        rgb_proj_param_ids = set(id(p) for p in self.heads.rgb_projector.parameters())
        
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                assert id(param) not in rgb_param_ids, \
                    "【禁止项】optimizer 不能包含 rgb_encoder 参数"
                assert id(param) not in rgb_proj_param_ids, \
                    "【禁止项】optimizer 不能包含 rgb_projector 参数"
    
    def _create_optimizer(self):
        """创建优化器（不包含 RGB 参数）"""
        # 收集可训练参数
        params = []
        
        # CSI Encoder 参数
        params.append({
            'params': self.csi_encoder.parameters(),
            'lr': float(self.cfg['training']['lr'])
        })
        
        # Heads 可训练参数（不包括 RGB Projector）
        params.append({
            'params': self.heads.get_trainable_params(),
            'lr': float(self.cfg['training']['lr']) * 10  # heads 使用更大的学习率
        })
        
        optimizer = AdamW(
            params,
            weight_decay=float(self.cfg['training']['weight_decay'])
        )
        
        return optimizer
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        warmup_epochs = int(self.cfg['training']['warmup_epochs'])
        total_epochs = int(self.cfg['training']['epochs'])
        
        # Warmup 调度器
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        
        # Cosine 调度器
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=1e-6
        )
        
        # 组合调度器
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
        
        return scheduler
    
    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个 epoch
        
        训练流程：
        1. CSI forward -> f_csi
        2. RGB forward (no_grad) -> f_rgb
        3. Project: z_csi, z_rgb
        4. Compute losses
        5. backward (only CSI graph)
        6. optimizer.step()
        7. momentum_update
        8. queue.enqueue(z_rgb.detach())
        
        Returns:
            metrics: 训练指标字典
        """
        self.csi_encoder.train()
        self.heads.train()
        # RGB Encoder 始终在 eval 模式（不更新 BN 统计量）
        self.rgb_encoder.eval()
        
        total_loss = 0.0
        total_con_loss = 0.0
        total_cls_loss = 0.0
        total_reg_loss = 0.0
        correct = 0
        total = 0
        
        epoch_start_time = time.time()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # 获取数据
            csi = batch['csi'].to(self.device)  # [B, T, S, A]
            skeleton = batch['skeleton'].to(self.device)  # [B, T, J, 3]
            labels = batch['label'].to(self.device)  # [B]
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 1. CSI forward -> fused feature
            f_csi = self.csi_encoder(csi)  # [B, fusion_dim]
            
            # 2. RGB forward (必须使用 no_grad)
            # 【关键约束】避免 RGB 建立计算图
            with torch.no_grad():
                f_rgb = self.rgb_encoder(skeleton)  # [B, fusion_dim]
            
            # 3. Project
            z_csi = self.heads.project_csi(f_csi)  # [B, projector_dim]
            with torch.no_grad():
                z_rgb = self.heads.project_rgb(f_rgb)  # [B, projector_dim]
            
            # 4. Compute losses
            # 4.1 对比损失
            queue_keys = self.queue.get_queue() if not self.queue.is_empty() else None
            queue_labels = self.queue.get_labels() if not self.queue.is_empty() else None
            
            loss_con = self.contrastive_loss(
                z_csi, z_rgb, labels, queue_keys, queue_labels
            )
            
            # 4.2 分类损失
            logits = self.heads.classify(f_csi)  # [B, num_classes]
            loss_cls = self.classification_loss(logits, labels)
            
            # 4.3 回归损失（辅助任务）
            # 对骨架进行时间池化，得到平均骨架
            skeleton_pooled = skeleton.mean(dim=1)  # [B, J, 3]
            skeleton_flat = skeleton_pooled.reshape(skeleton_pooled.shape[0], -1)  # [B, J*3]
            joints_pred = self.heads.regress(f_csi)  # [B, J*3]
            loss_reg = self.regression_loss(joints_pred, skeleton_flat)
            
            # 4.4 总损失
            lambda_con = float(self.cfg['training']['lambda_con'])
            lambda_cls = float(self.cfg['training']['lambda_cls'])
            lambda_reg = float(self.cfg['training']['lambda_reg'])
            
            loss = lambda_con * loss_con + lambda_cls * loss_cls + lambda_reg * loss_reg
            
            # 5. Backward (only CSI graph)
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.csi_encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.heads.get_trainable_params(), max_norm=1.0)
            
            # 6. Optimizer step
            self.optimizer.step()
            
            # 7. Momentum update RGB Projector (必须在 optimizer.step() 之后)
            # 注意：由于 RGB Encoder 和 CSI Encoder 结构不同，
            # 我们只对 Projector 进行 EMA 更新
            self._momentum_update_projector()
            
            # 8. Queue enqueue (必须使用 detach)
            # 【关键约束】只存储 RGB keys
            self.queue.enqueue(z_rgb.detach(), labels)
            
            # 统计
            total_loss += loss.item()
            total_con_loss += loss_con.item()
            total_cls_loss += loss_cls.item()
            total_reg_loss += loss_reg.item()
            
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # 样本可视化（第一个批次且每10个epoch一次）
            if batch_idx == 0:
                self._log_samples_to_tensorboard(
                    self.current_epoch, csi, skeleton, labels, logits
                )
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
        
        # 更新学习率
        self.scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        
        # 计算平均指标
        num_batches = len(self.train_loader)
        metrics = {
            'train_loss': total_loss / num_batches,
            'train_con_loss': total_con_loss / num_batches,
            'train_cls_loss': total_cls_loss / num_batches,
            'train_reg_loss': total_reg_loss / num_batches,
            'train_acc': 100. * correct / total,
            'lr': self.optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time,
            'queue_size': self.queue.size()
        }
        
        return metrics
    
    @torch.no_grad()
    def _momentum_update_projector(self):
        """动量更新 RGB Projector"""
        m = float(self.cfg['training']['ema_m'])
        for p_rgb, p_csi in zip(
            self.heads.rgb_projector.parameters(),
            self.heads.csi_projector.parameters()
        ):
            p_rgb.data.mul_(m).add_(p_csi.data, alpha=1 - m)
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        验证
        
        【关键】验证时只使用 CSI Encoder + Classifier
        不使用 RGB、Queue、Regressor、Projector
        
        Returns:
            metrics: 验证指标字典
        """
        self.csi_encoder.eval()
        self.heads.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            csi = batch['csi'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 只使用 CSI Encoder + Classifier
            f_csi = self.csi_encoder(csi)
            logits = self.heads.classify(f_csi)
            
            loss = self.classification_loss(logits, labels)
            total_loss += loss.item()
            
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        accuracy = 100. * correct / total
        
        # 计算 F1 分数
        from sklearn.metrics import f1_score
        f1_macro = f1_score(all_labels, all_preds, average='macro') * 100
        f1_weighted = f1_score(all_labels, all_preds, average='weighted') * 100
        
        metrics = {
            'val_loss': total_loss / len(self.val_loader),
            'val_acc': accuracy,
            'val_f1_macro': f1_macro,
            'val_f1_weighted': f1_weighted
        }
        
        return metrics
    
    def train(self, num_epochs: Optional[int] = None):
        """
        完整训练流程
        
        Args:
            num_epochs: 训练轮数，默认使用配置中的值
        """
        if num_epochs is None:
            num_epochs = int(self.cfg['training']['epochs'])
        
        self.logger.log(f"\n开始训练，共 {num_epochs} 轮")
        self.logger.log("=" * 60)
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_metrics = self.train_epoch()
            
            # 验证
            val_metrics = self.validate()
            
            # 合并指标
            metrics = {
                **train_metrics, 
                **val_metrics, 
                'epoch': epoch,
                'timestamp': datetime.now().isoformat()
            }
            self.training_history.append(metrics)
            
            # 记录日志
            self.logger.log(f"\nEpoch {epoch}/{num_epochs-1}:")
            self.logger.log(f"  训练损失: {train_metrics['train_loss']:.4f} "
                          f"(对比: {train_metrics['train_con_loss']:.4f}, "
                          f"分类: {train_metrics['train_cls_loss']:.4f}, "
                          f"回归: {train_metrics['train_reg_loss']:.4f})")
            self.logger.log(f"  训练准确率: {train_metrics['train_acc']:.2f}%")
            self.logger.log(f"  验证损失: {val_metrics['val_loss']:.4f}")
            self.logger.log(f"  验证准确率: {val_metrics['val_acc']:.2f}%")
            self.logger.log(f"  验证 F1 (Macro): {val_metrics['val_f1_macro']:.2f}%")
            self.logger.log(f"  学习率: {train_metrics['lr']:.6f}")
            self.logger.log(f"  队列大小: {train_metrics['queue_size']}")
            self.logger.log(f"  耗时: {train_metrics['epoch_time']:.1f}s")
            
            # 记录指标到文件
            self.logger.log_metrics(metrics)
            
            # 记录指标到 TensorBoard
            self._log_to_tensorboard(epoch, train_metrics, val_metrics)
            
            # 保存最佳模型（如果配置启用）
            if self.cfg.get('checkpoint', {}).get('save_best', True):
                if val_metrics['val_acc'] > self.best_metric:
                    self.best_metric = val_metrics['val_acc']
                    self.save_checkpoint('best.pth')
                    self.logger.log(f"  ★ 新最佳模型! 准确率: {self.best_metric:.2f}%")
            
            # 早停检查
            if self.early_stopping_enabled:
                should_stop = self._check_early_stopping(val_metrics)
                if should_stop:
                    self.logger.log(f"\n早停触发! 验证指标在 {self.early_stopping_patience} 轮内未改善")
                    self.logger.log(f"最佳 {self.early_stopping_monitor}: {self.best_early_stop_metric:.4f}")
                    break
            
            # 定期保存（根据配置的间隔）
            save_interval = self.cfg.get('checkpoint', {}).get('save_interval', 10)
            if save_interval > 0 and (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f'epoch_{epoch}.pth')
        
        # 保存最终模型（如果配置启用）
        if self.cfg.get('checkpoint', {}).get('save_final', True):
            self.save_checkpoint('final.pth')
        
        # 关闭 TensorBoard
        self.close_tensorboard()
        
        self.logger.log("\n" + "=" * 60)
        self.logger.log("训练完成!")
        self.logger.log(f"最佳验证准确率: {self.best_metric:.2f}%")
        if self.early_stopping_enabled:
            self.logger.log(f"早停状态: {'已触发' if should_stop else '未触发'}")
        self.logger.log("=" * 60)
    
    def _check_early_stopping(self, val_metrics: Dict) -> bool:
        """
        检查是否应该早停
        
        Args:
            val_metrics: 验证指标字典
            
        Returns:
            bool: 是否应该停止训练
        """
        # 获取当前监控的指标值
        current_metric = val_metrics.get(self.early_stopping_monitor, 0.0)
        
        # 判断是否改善
        if 'acc' in self.early_stopping_monitor or 'f1' in self.early_stopping_monitor:
            # 准确率或 F1：越大越好
            improved = current_metric > (self.best_early_stop_metric + self.early_stopping_min_delta)
            if improved:
                self.best_early_stop_metric = current_metric
                self.early_stopping_counter = 0
                self.logger.log(f"  早停指标改善: {self.early_stopping_monitor}={current_metric:.4f}")
            else:
                self.early_stopping_counter += 1
                self.logger.log(f"  早停计数器: {self.early_stopping_counter}/{self.early_stopping_patience}")
        else:
            # 损失：越小越好
            improved = current_metric < (self.best_early_stop_metric - self.early_stopping_min_delta)
            if improved:
                self.best_early_stop_metric = current_metric
                self.early_stopping_counter = 0
                self.logger.log(f"  早停指标改善: {self.early_stopping_monitor}={current_metric:.4f}")
            else:
                self.early_stopping_counter += 1
                self.logger.log(f"  早停计数器: {self.early_stopping_counter}/{self.early_stopping_patience}")
        
        # 判断是否应该停止
        return self.early_stopping_counter >= self.early_stopping_patience
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        save_dir = self.cfg.get('output_folder', './checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'best_metric': self.best_metric,
            'csi_encoder': self.csi_encoder.state_dict(),
            'rgb_encoder': self.rgb_encoder.state_dict(),
            'heads': self.heads.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'queue': self.queue.state_dict(),
            'training_history': self.training_history,
            'config': self.cfg
        }
        
        filepath = os.path.join(save_dir, filename)
        torch.save(checkpoint, filepath)
        self.logger.log(f"  检查点已保存: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_metric = checkpoint['best_metric']
        self.csi_encoder.load_state_dict(checkpoint['csi_encoder'])
        self.rgb_encoder.load_state_dict(checkpoint['rgb_encoder'])
        self.heads.load_state_dict(checkpoint['heads'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.queue.load_state_dict(checkpoint['queue'])
        # momentum_updater 已移除，不再加载
        self.training_history = checkpoint['training_history']
        
        self.logger.log(f"检查点已加载: {filepath}")
        self.logger.log(f"  从 epoch {self.current_epoch} 继续训练")
        self.logger.log(f"  最佳准确率: {self.best_metric:.2f}%")
    
    def _log_to_tensorboard(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """记录指标到 TensorBoard"""
        
        # ===== 1. 损失函数监控 =====
        # 总损失
        self.writer.add_scalar('Loss/Train_Total', train_metrics['train_loss'], epoch)
        self.writer.add_scalar('Loss/Validation_Total', val_metrics['val_loss'], epoch)
        
        # 分解损失（对比、分类、回归）
        self.writer.add_scalar('Loss/Train_Contrastive', train_metrics['train_con_loss'], epoch)
        self.writer.add_scalar('Loss/Train_Classification', train_metrics['train_cls_loss'], epoch)
        self.writer.add_scalar('Loss/Train_Regression', train_metrics['train_reg_loss'], epoch)
        
        # 损失比例（帮助调试损失权重）
        total_loss = train_metrics['train_loss']
        if total_loss > 0:
            self.writer.add_scalar('Loss_Ratio/Contrastive_Ratio', 
                                 train_metrics['train_con_loss'] / total_loss, epoch)
            self.writer.add_scalar('Loss_Ratio/Classification_Ratio', 
                                 train_metrics['train_cls_loss'] / total_loss, epoch)
            self.writer.add_scalar('Loss_Ratio/Regression_Ratio', 
                                 train_metrics['train_reg_loss'] / total_loss, epoch)
        
        # ===== 2. 准确率与 F1 分数 =====
        self.writer.add_scalar('Accuracy/Train', train_metrics['train_acc'], epoch)
        self.writer.add_scalar('Accuracy/Validation', val_metrics['val_acc'], epoch)
        self.writer.add_scalar('F1/Validation_Macro', val_metrics['val_f1_macro'], epoch)
        
        # 准确率差异（过拟合检测）
        acc_gap = train_metrics['train_acc'] - val_metrics['val_acc']
        self.writer.add_scalar('Overfitting/Accuracy_Gap', acc_gap, epoch)
        
        # ===== 3. 学习率监控 =====
        self.writer.add_scalar('Learning_Rate/Current_LR', train_metrics['lr'], epoch)
        
        # ===== 4. 模型参数监控 =====
        # CSI Encoder 参数直方图
        for name, param in self.csi_encoder.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f'Parameters/CSI_Encoder/{name}', param.data, epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f'Gradients/CSI_Encoder/{name}', param.grad, epoch)
        
        # Heads 参数直方图（只记录可训练的）
        for name, param in self.heads.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f'Parameters/Heads/{name}', param.data, epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f'Gradients/Heads/{name}', param.grad, epoch)
        
        # ===== 5. 梯度流监控 =====
        # 计算梯度范数
        csi_grad_norm = 0.0
        heads_grad_norm = 0.0
        
        for param in self.csi_encoder.parameters():
            if param.grad is not None:
                csi_grad_norm += param.grad.data.norm(2).item() ** 2
        csi_grad_norm = csi_grad_norm ** 0.5
        
        for param in self.heads.get_trainable_params():
            if param.grad is not None:
                heads_grad_norm += param.grad.data.norm(2).item() ** 2
        heads_grad_norm = heads_grad_norm ** 0.5
        
        self.writer.add_scalar('Gradients/CSI_Encoder_Norm', csi_grad_norm, epoch)
        self.writer.add_scalar('Gradients/Heads_Norm', heads_grad_norm, epoch)
        
        # ===== 6. 内存队列监控 =====
        self.writer.add_scalar('Queue/Size', train_metrics['queue_size'], epoch)
        self.writer.add_scalar('Queue/Utilization', 
                             train_metrics['queue_size'] / self.queue.queue_size, epoch)
        
        # ===== 7. 训练效率监控 =====
        self.writer.add_scalar('Time/Epoch_Duration', train_metrics['epoch_time'], epoch)
        
        # ===== 8. 模型健康度监控 =====
        # 检查是否有 NaN 或 Inf
        has_nan = any(torch.isnan(param).any() for param in self.csi_encoder.parameters())
        has_inf = any(torch.isinf(param).any() for param in self.csi_encoder.parameters())
        self.writer.add_scalar('Health/Has_NaN', float(has_nan), epoch)
        self.writer.add_scalar('Health/Has_Inf', float(has_inf), epoch)
        
        # 参数更新幅度（监控学习是否停滞）
        if hasattr(self, '_prev_params'):
            param_change = 0.0
            param_count = 0
            for (name, param), prev_param in zip(self.csi_encoder.named_parameters(), self._prev_params):
                if param.requires_grad:
                    change = (param.data - prev_param).norm().item()
                    param_change += change
                    param_count += 1
            if param_count > 0:
                avg_param_change = param_change / param_count
                self.writer.add_scalar('Health/Avg_Parameter_Change', avg_param_change, epoch)
        
        # 保存当前参数用于下次比较
        self._prev_params = [param.data.clone() for param in self.csi_encoder.parameters() if param.requires_grad]
        
        # ===== 9. EMA 动量监控 =====
        # 监控 RGB Projector 的 EMA 更新
        if hasattr(self, '_prev_rgb_params'):
            rgb_change = 0.0
            rgb_count = 0
            for param, prev_param in zip(self.heads.rgb_projector.parameters(), self._prev_rgb_params):
                change = (param.data - prev_param).norm().item()
                rgb_change += change
                rgb_count += 1
            if rgb_count > 0:
                avg_rgb_change = rgb_change / rgb_count
                self.writer.add_scalar('EMA/RGB_Projector_Change', avg_rgb_change, epoch)
        
        # 保存当前 RGB 参数
        self._prev_rgb_params = [param.data.clone() for param in self.heads.rgb_projector.parameters()]
        
        # 刷新缓冲区
        self.writer.flush()
    
    def _log_samples_to_tensorboard(self, epoch: int, csi_batch: torch.Tensor, 
                                   skeleton_batch: torch.Tensor, labels: torch.Tensor, 
                                   predictions: torch.Tensor, max_samples: int = 4):
        """记录样本可视化到 TensorBoard（可选功能）"""
        if epoch % 10 != 0:  # 每 10 个 epoch 记录一次，避免过多数据
            return
            
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            
            batch_size = min(max_samples, csi_batch.size(0))
            
            for i in range(batch_size):
                # CSI 数据可视化（显示幅值热图）
                csi_sample = csi_batch[i].cpu().numpy()  # [T, Subcarriers, Antennas]
                
                # 创建 CSI 热图（时间 x 子载波，对天线维度求平均）
                csi_heatmap = np.mean(csi_sample, axis=2)  # [T, Subcarriers]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # CSI 热图
                im1 = ax1.imshow(csi_heatmap.T, aspect='auto', cmap='viridis')
                ax1.set_title(f'CSI Sample {i} (Label: {labels[i].item()}, Pred: {predictions[i].argmax().item()})')
                ax1.set_xlabel('Time Steps')
                ax1.set_ylabel('Subcarriers')
                plt.colorbar(im1, ax=ax1)
                
                # 骨架数据可视化（显示关节轨迹）
                skeleton_sample = skeleton_batch[i].cpu().numpy()  # [T, J, 3]
                
                # 选择几个关键关节进行可视化
                key_joints = [0, 1, 2, 5, 6, 9, 10]  # 头部、肩膀、手腕、膝盖等
                for j, joint_idx in enumerate(key_joints):
                    if joint_idx < skeleton_sample.shape[1]:
                        joint_traj = skeleton_sample[:, joint_idx, :]  # [T, 3]
                        ax2.plot(joint_traj[:, 0], joint_traj[:, 1], 
                                label=f'Joint {joint_idx}', alpha=0.7)
                
                ax2.set_title(f'Skeleton Trajectory Sample {i}')
                ax2.set_xlabel('X Coordinate')
                ax2.set_ylabel('Y Coordinate')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # 转换为 TensorBoard 图像
                fig.canvas.draw()
                img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                
                # 记录到 TensorBoard
                self.writer.add_image(f'Samples/Sample_{i}', img_array, epoch, dataformats='HWC')
                
                plt.close(fig)
                
        except ImportError:
            # matplotlib 不可用时跳过可视化
            pass
        except Exception as e:
            # 其他错误时记录但不中断训练
            self.logger.log(f"样本可视化失败: {e}")
    
    def close_tensorboard(self):
        """关闭 TensorBoard writer"""
        if hasattr(self, 'writer'):
            self.writer.close()
    
    @torch.no_grad()
    def evaluate(self, test_loader=None) -> Dict[str, float]:
        """
        评估（推理模式）
        
        【关键约束】
        - 只使用 CSI Encoder + Classifier
        - 不使用 RGB、Queue、Regressor、Projector
        
        Args:
            test_loader: 测试数据加载器，默认使用验证集
        
        Returns:
            metrics: 评估指标
        """
        if test_loader is None:
            test_loader = self.val_loader
        
        self.csi_encoder.eval()
        self.heads.classifier.eval()
        
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        self.logger.log("\n开始评估（推理模式）")
        self.logger.log("【约束验证】只使用 CSI Encoder + Classifier")
        
        for batch in tqdm(test_loader, desc='Evaluation'):
            csi = batch['csi'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 【推理】只使用 CSI Encoder + Classifier
            f_csi = self.csi_encoder(csi)
            logits = self.heads.classify(f_csi)
            
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        accuracy = 100. * correct / total
        
        # 计算混淆矩阵和每类准确率
        from sklearn.metrics import confusion_matrix, classification_report, f1_score
        cm = confusion_matrix(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro') * 100
        f1_weighted = f1_score(all_labels, all_preds, average='weighted') * 100
        
        metrics = {
            'test_acc': accuracy,
            'test_f1_macro': f1_macro,
            'test_f1_weighted': f1_weighted,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'labels': all_labels
        }
        
        self.logger.log(f"\n评估结果:")
        self.logger.log(f"  准确率: {accuracy:.2f}%")
        self.logger.log(f"  F1 (Macro): {f1_macro:.2f}%")
        self.logger.log(f"  F1 (Weighted): {f1_weighted:.2f}%")
        
        return metrics


# 测试代码
if __name__ == '__main__':
    print("Trainer 模块测试...")
    print("请使用完整的训练脚本进行测试")
