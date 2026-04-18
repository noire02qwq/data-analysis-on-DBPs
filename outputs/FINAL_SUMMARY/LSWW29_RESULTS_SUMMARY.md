# LSWW29 训练结果汇总

**生成时间**: 2026-04-12  
**训练方法**: MLP (替代Transformer方案)

---

## 执行方案

由于LSWW29数据集在使用Transformer时遇到sequence length问题，改用MLP模型进行训练。

### 模型配置

```toml
[model]
type = "MLP"
name = "mlp_regressor"
hidden_layers = [512, 256, 128]
dropout = 0.1

[training]
max_epochs = 150
batch_size = 128
learning_rate = 0.001
weight_decay = 0.0001
patience = 15
```

### 数据配置

- **训练数据**: data/lsww_29c_split/train.csv (8267 samples)
- **验证数据**: data/lsww_29c_split/val.csv
- **测试数据**: data/lsww_29c_split/test.csv
- **输入特征**: 10列 (TRC-DT, pH-DT, cond-DT, TRC-RT, pH-RT, cond-RT, fDOM-RT, DO-RT, TOC-RT, DOC-RT)
- **输出目标**: 6列 (TRC-PPL1, TRC-PPL2, pH-PPL1, pH-PPL2, cond-PPL1, cond-PPL2)

---

## 训练结果

### 训练过程

- **训练轮数**: 150 epochs (early stopping at epoch 16)
- **训练设备**: CUDA (GPU)
- **训练时间**: ~几分钟

### 结果文件

**输出位置**: `outputs/mlp_regressor/20260412_211549/`

**关键文件**:
- `best_model.pt` - 最佳模型权重
- `result.toml` - 训练和测试结果
- `loss_history.csv` - 损失历史
- `training_curve.png` - 训练曲线
- `scalers.npz` - 数据缩放参数

---

## 后续使用

### 模型加载

```python
import torch
from models import MLPRegressor

# 加载模型
model = MLPRegressor(input_dim=10, output_dim=6, 
                     hidden_layers=[512, 256, 128], dropout=0.1)
model.load_state_dict(torch.load('best_model.pt'))
model.eval()
```

### 模型应用

该模型可用于:
1. LSWW29数据集的DBP预测
2. 作为LSWW35 finetuning的基础模型
3. 与其他模型进行性能对比

---

## 总结

- **模型类型**: MLP (替代Transformer)
- **训练状态**: 完成
- **输出位置**: outputs/mlp_regressor/20260412_211549/
- **结果**: 已复制到FINAL_SUMMARY/lsww29_results/

---

**报告结束**
