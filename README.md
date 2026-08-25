# 基于跨模态动量对比对齐的 CSI-only 人体活动识别

## 研究目标

本项目实现了一套基于跨模态动量对比学习的人体活动识别系统，使用 MMFi 数据集。

**核心思想**：在训练阶段，利用 RGB 骨架提供的高质量动作语义空间，通过 MoCo-style 跨模态对比学习，引导 CSI 学习鲁棒且具判别性的动作表示；在测试阶段，仅依赖 CSI 完成活动识别。



A PyTorch research prototype for CSI-based human activity recognition. The framework uses synchronized RGB skeletons as privileged supervision during training, while inference relies exclusively on CSI signals.

Status: Research prototype. Training, evaluation, CSI-only inference, and architectural constraint tests are implemented; final benchmark performance is still under validation.



## 方法概要

### 整体框架

```
训练阶段:
CSI 数据 → CSI Encoder (Dual-Stream) → CSI Feature → Projector → z_csi (Query)
                                              ↓
                                         Classifier → 分类损失
                                              ↓
                                         Regressor → 回归损失（辅助）
                                              
RGB 骨架 → RGB Encoder (Momentum) → RGB Feature → Projector → z_rgb (Key)
                    ↑                                              ↓
                    └──────── EMA 更新 ←──────────────────────────┘
                                                                   ↓
                                                            Memory Queue
                                                                   ↓
                                                            对比损失

推理阶段:
CSI 数据 → CSI Encoder → CSI Feature → Classifier → 活动类别
```

### 关键技术

1. **CSI 双流编码器（DualConFi 风格）**
   - Channel Stream: 建模子载波/天线间的频域相关性
   - Temporal Stream: 建模动作的时序动态
   - Fusion Module: 后期融合产生统一表征

2. **RGB 动量编码器（MoCo 风格）**
   - 参数通过 EMA 从 CSI 编码器更新
   - 不参与梯度反向传播
   - 提供稳定的语义锚点

3. **监督对比学习（SupCon 风格）**
   - 正样本：同一实例的 CSI-RGB 对
   - 负样本：Memory Queue 中的历史 RGB keys
   - 支持 Hard Negative 采样

4. **CSI 专用增强（DeepCRF 风格）**
   - 保持动作语义的增强策略
   - 高斯噪声、时间遮挡、子载波遮挡等

## 项目结构

```
├── datasets/
│   └── mmfi_dataset.py          # MMFi 数据集包装器和 CSI 增强
├── models/
│   ├── csi_encoder.py           # CSI 双流编码器
│   ├── rgb_encoder.py           # RGB 动量编码器
│   ├── momentum.py              # EMA 动量更新机制
│   └── heads.py                 # 投影头、分类头、回归头
├── losses/
│   └── supcon.py                # 监督对比损失
├── utils/
│   └── queue.py                 # 内存队列（存储 RGB keys）
├── trainer/
│   └── trainer.py               # 训练、验证、评估逻辑
├── configs/
│   └── default.yaml             # 默认配置文件
├── notebooks/
│   └── experiment_demo.ipynb    # 实验演示 notebook
├── tests/                       # 单元测试
│   ├── dataset_test.py
│   ├── no_rgb_grad_test.py
│   ├── optimizer_excludes_rgb_test.py
│   ├── ema_update_test.py
│   ├── queue_test.py
│   └── inference_test.py
├── train.py                     # 训练脚本
├── evaluate.py                  # 评估脚本
├── inference.py                 # 推理脚本
└── run_tests.py                 # 测试运行脚本
```

## 运行说明

### 环境安装

```bash
# 创建虚拟环境
conda create -n csi_har python=3.8
conda activate csi_har

# 安装依赖
pip install -r requirements.txt
```

### 配置

修改 `configs/default.yaml` 中的数据路径：

```yaml
dataset:
  root: "/path/to/MMFi_Dataset"  # 修改为实际路径
```

### 训练

```bash
# 使用默认配置训练
python train.py --config configs/default.yaml

# 从检查点恢复训练
python train.py --config configs/default.yaml --resume checkpoints/best.pth
```

### TensorBoard 可视化

训练过程中会自动记录指标到 TensorBoard，可以实时查看训练效果：

```bash
# 启动 TensorBoard（在新的终端窗口中运行）
tensorboard --logdir=./logs/tensorboard

# 或者指定端口
tensorboard --logdir=./logs/tensorboard --port=6006

# 然后在浏览器中打开
# http://localhost:6006
```

#### 监控指标详解

**📊 1. 损失函数监控**
- `Loss/Train_Total` - 训练总损失，观察模型收敛情况
- `Loss/Validation_Total` - 验证总损失，评估泛化能力
- `Loss/Train_Contrastive` - 对比损失，监控跨模态对齐效果
- `Loss/Train_Classification` - 分类损失，监控主任务性能
- `Loss/Train_Regression` - 回归损失，监控辅助任务性能
- `Loss_Ratio/*` - 各损失占比，帮助调试损失权重配置

**📈 2. 准确率与性能**
- `Accuracy/Train` - 训练准确率
- `Accuracy/Validation` - 验证准确率
- `F1/Validation_Macro` - 宏平均 F1 分数（不平衡数据集重要指标）
- `Overfitting/Accuracy_Gap` - 训练验证准确率差异（过拟合检测）

**⚙️ 3. 学习率监控**
- `Learning_Rate/Current_LR` - 当前学习率，确保调度器正常工作

**🔍 4. 模型参数监控**
- `Parameters/CSI_Encoder/*` - CSI 编码器参数分布直方图
- `Parameters/Heads/*` - 头部模块参数分布直方图
- `Gradients/CSI_Encoder/*` - CSI 编码器梯度分布直方图
- `Gradients/Heads/*` - 头部模块梯度分布直方图

**📉 5. 梯度流监控**
- `Gradients/CSI_Encoder_Norm` - CSI 编码器梯度范数
- `Gradients/Heads_Norm` - 头部模块梯度范数
- 帮助检测梯度爆炸/消失问题

**🗃️ 6. 内存队列监控**
- `Queue/Size` - 当前队列大小
- `Queue/Utilization` - 队列利用率（当前大小/最大容量）

**⏱️ 7. 训练效率监控**
- `Time/Epoch_Duration` - 每个 epoch 训练时间

**🏥 8. 模型健康度监控**
- `Health/Has_NaN` - 检测参数中是否有 NaN
- `Health/Has_Inf` - 检测参数中是否有无穷大
- `Health/Avg_Parameter_Change` - 平均参数更新幅度（检测学习停滞）

**🔄 9. EMA 动量监控**
- `EMA/RGB_Projector_Change` - RGB 投影头 EMA 更新幅度

**🖼️ 10. 样本可视化（可选）**
- `Samples/Sample_*` - 每 10 个 epoch 记录训练样本可视化
  - CSI 数据热图（时间 x 子载波）
  - 骨架关节轨迹图
  - 真实标签 vs 预测标签对比

#### 使用技巧

**实时监控建议：**
1. **损失趋势**：训练损失应稳定下降，验证损失不应持续上升
2. **过拟合检测**：关注 `Overfitting/Accuracy_Gap`，差异过大说明过拟合
3. **梯度健康**：梯度范数应在合理范围内（通常 0.1-10）
4. **参数更新**：`Health/Avg_Parameter_Change` 过小说明学习停滞
5. **队列状态**：队列利用率应逐渐增长到 100%

**调试指南：**
- 📉 **损失不下降**：检查学习率、梯度范数、参数更新幅度
- 📈 **过拟合严重**：关注准确率差异，考虑增加正则化
- ⚠️ **梯度爆炸**：梯度范数过大（>100），需要梯度裁剪
- 🐌 **学习停滞**：参数更新幅度过小，考虑提高学习率
- 💥 **数值不稳定**：出现 NaN/Inf，检查损失函数和数据预处理

**界面操作：**
- 左侧面板可以选择显示/隐藏不同的指标
- 使用平滑功能查看趋势（Smoothing 滑块）
- 可以同时比较多次实验的结果
- 支持下载图表为 PNG/SVG 格式
- 使用正则表达式过滤指标名称

### 评估

```bash
python evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pth
```

### 推理

```bash
# 单个文件推理
python inference.py --config configs/default.yaml --checkpoint checkpoints/best.pth --input path/to/csi.npy

# 批量推理
python inference.py --config configs/default.yaml --checkpoint checkpoints/best.pth --input path/to/csi_dir/
```

### 运行测试

```bash
python run_tests.py
```

## 实验环境

- Python >= 3.8
- PyTorch >= 1.8.0
- TensorBoard >= 2.8.0
- NumPy >= 1.19.0
- SciPy >= 1.6.0
- PyYAML >= 5.4.0
- scikit-learn >= 0.24.0
- tqdm >= 4.60.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0

## 重要约束

### 训练阶段

- RGB 编码器参数被冻结，不参与梯度反向传播
- RGB 编码器参数通过 CSI 编码器参数的 EMA 更新
- 内存队列只存储 RGB keys，不存储 CSI keys
- 总损失 = λ₁ × 对比损失 + λ₂ × 分类损失 + λ₃ × 回归损失

### 推理阶段

- **严格禁止**使用 RGB 数据
- **严格禁止**使用内存队列
- **严格禁止**使用回归头
- **严格禁止**使用投影头
- 只使用 CSI 编码器 + 分类头

## 超参数推荐与调优指南

### 基础超参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| batch_size | 64 | OOM 时降至 32 |
| epochs | 100-200 | 训练轮数（启用早停后可设置更大值） |
| lr | 1e-4 | 编码器学习率 |
| weight_decay | 1e-4 | 权重衰减 |
| warmup_epochs | 5 | 预热轮数 |
| temperature | 0.07 | 对比学习温度 |
| ema_m | 0.999 | EMA 动量因子 |
| queue_size | 4096 | 内存队列大小 |

### 损失权重调优（重要！）

| 参数 | 推荐值 | 调优建议 |
|------|--------|----------|
| lambda_con | 1.0 | 对比损失权重，保持为 1.0 |
| lambda_cls | 1.0 | 分类损失权重，主任务，保持为 1.0 |
| lambda_reg | 0.1-0.3 | **回归损失权重，建议降低到 0.1-0.3** |

**⚠️ 重要提示：** 回归损失数值通常较大，如果权重过高（如 0.5）会主导优化过程，影响分类性能。建议：
1. 初始设置为 0.1
2. 通过 TensorBoard 监控 `Loss_Ratio/*` 指标
3. 确保回归损失占比不超过总损失的 30%
4. 如果验证准确率低，尝试进一步降低 lambda_reg

### 正则化与防过拟合

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| dropout | 0.1 | Dropout 比例（0 表示不使用） |
| label_smoothing | 0.1 | 标签平滑（减少过拟合） |
| regression_loss_type | "smooth_l1" | 回归损失类型，推荐 smooth_l1 或 huber |

**回归损失类型选择：**
- `mse`: 均方误差，对离群点敏感
- `l1`: 绝对值误差，对离群点鲁棒
- `smooth_l1`: **推荐**，结合 MSE 和 L1 优点
- `huber`: 类似 smooth_l1，对离群点更鲁棒

### 早停配置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| early_stopping.enabled | true | 是否启用早停 |
| early_stopping.patience | 15-20 | 容忍轮数 |
| early_stopping.min_delta | 0.001 | 最小改进阈值 |
| early_stopping.monitor | "val_acc" | 监控指标 |

**早停策略：**
- 监控 `val_acc`（验证准确率）或 `val_loss`（验证损失）
- patience 设置为 15-20 轮，避免过早停止
- 启用早停后，epochs 可以设置为较大值（如 200）

### 调优流程建议

**第一阶段：基线训练**
1. 使用默认配置训练 20-30 轮
2. 通过 TensorBoard 观察：
   - 损失比例是否合理
   - 是否出现过拟合
   - 梯度是否正常

**第二阶段：损失权重调整**
1. 如果回归损失占比过高（>40%）：
   - 降低 `lambda_reg` 到 0.1-0.2
2. 如果训练验证准确率差距大（>10%）：
   - 启用 `label_smoothing: 0.1`
   - 启用 `dropout: 0.1`

**第三阶段：早停与长期训练**
1. 启用早停机制
2. 设置 epochs 为 200
3. 让模型自动找到最佳停止点


## 结果复现

1. 下载 MMFi 数据集并解压
2. 修改配置文件中的数据路径
3. 运行训练脚本
4. 使用最佳检查点进行评估

详细的实验演示请参考 `notebooks/experiment_demo.ipynb`。

## 引用

如果您使用了本代码，请引用：

```bibtex
@article{your_paper,
  title={Cross-modal Momentum Contrastive Learning for CSI-only Human Activity Recognition},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## 数据与第三方组件

本项目包含用于加载 MM-Fi 数据集的工具箱代码（`MMFi_dataset/`）。请遵循
[MM-Fi 项目页](https://ntu-aiot-lab.github.io/mm-fi)所述的数据访问、署名与使用条件；
该项目页将 MM-Fi 数据集标注为 CC BY-NC 4.0。仓库当前不对整体代码声明 MIT
许可证。


