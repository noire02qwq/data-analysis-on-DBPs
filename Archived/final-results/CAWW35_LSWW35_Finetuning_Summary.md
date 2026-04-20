# CAWW35 和 LSWW35 数据集 Fine-tuning 实验总结报告

## 实验概述

本实验对比了三种 Fine-tuning 方法在 CAWW35 和 LSWW35 两个新数据集上的表现：
- **Full Fine-tuning**: 全模型微调
- **Partial Fine-tuning**: 冻结底层，只微调顶层
- **Adapter**: 使用 Adapter 层进行参数高效微调

所有实验均基于从 CAWW29/LSWW29 预训练的 Transformer 模型。

---

## 实验结果汇总

### CAWW35 数据集

| Fine-tuning 方法 | Validation Loss | Test Loss | Best Epoch |
|-----------------|-----------------|-----------|------------|
| **Full** | **0.0737** | **0.2624** | 31 |
| Partial | 0.2079 | 0.3635 | 11 |
| Adapter | 0.3572 | 0.6458 | 45 |

### LSWW35 数据集

| Fine-tuning 方法 | Validation Loss | Test Loss | Best Epoch |
|-----------------|-----------------|-----------|------------|
| **Full** | **0.2531** | 0.8603 | 28 |
| **Partial** | 0.3238 | **0.3812** | 4 |
| Adapter | 0.3186 | 0.4584 | 1 |

---

## 关键发现与分析

### 1. CAWW35 数据集表现

**Full Fine-tuning 明显优于其他方法：**
- Validation Loss: 0.0737 (比 Partial 低 64.6%，比 Adapter 低 79.4%)
- Test Loss: 0.2624 (比 Partial 低 27.8%，比 Adapter 低 59.4%)
- 收敛速度适中（31 epochs）

**分析：**
- CAWW35 数据分布与预训练数据（CAWW29）有一定差异
- 全模型微调能够充分适应新数据的特征分布
- Adapter 层可能限制了模型的表达能力

### 2. LSWW35 数据集表现

**Partial Fine-tuning 在 Test Loss 上表现最佳：**
- Test Loss: 0.3812 (比 Full 低 55.7%)
- 但 Validation Loss (0.3238) 不如 Full Fine-tuning (0.2531)

**Full Fine-tuning 在 Validation Loss 上表现最佳：**
- Validation Loss: 0.2531 (比 Partial 低 21.8%)
- 但 Test Loss (0.8603) 明显较差，存在过拟合风险

**Adapter 表现平稳：**
- Validation Loss: 0.3186
- Test Loss: 0.4584

**分析：**
- LSWW35 数据集可能存在分布偏移或噪声
- Full Fine-tuning 在验证集上表现好但在测试集上差，说明可能过拟合
- Partial Fine-tuning 通过冻结底层参数，保持了预训练知识的泛化能力

### 3. 两种数据集的对比

| 特性 | CAWW35 | LSWW35 |
|-----|--------|--------|
| 最佳方法 | Full Fine-tuning | Partial Fine-tuning |
| 最佳 Test Loss | 0.2624 | 0.3812 |
| 数据质量 | 较高 | 可能存在噪声/偏移 |
| 过拟合风险 | 低 | 高（Full 方法） |

---

## 方法对比总结

### Full Fine-tuning
**优势：**
- 模型容量最大，能适应复杂的新任务
- 在数据质量高、分布相似时表现最佳

**劣势：**
- 需要更多训练数据
- 容易过拟合，特别是数据量小或分布差异大时
- 训练时间长

### Partial Fine-tuning
**优势：**
- 保留预训练知识，泛化能力强
- 在数据有噪声或分布偏移时表现稳健
- 训练速度快

**劣势：**
- 可能无法充分适应与预训练差异很大的新任务

### Adapter
**优势：**
- 参数高效，训练速度快
- 不改变原模型参数，便于多任务切换

**劣势：**
- 表达能力受限
- 在需要大幅调整的任务上表现不佳

---

## 建议与结论

### 针对不同数据集的建议

**CAWW35（高质量数据）：**
- ✅ 推荐使用 **Full Fine-tuning**
- 模型能够充分利用新数据，达到最佳性能

**LSWW35（可能存在噪声/偏移）：**
- ✅ 推荐使用 **Partial Fine-tuning**
- 在保持泛化能力的同时适应新任务
- ⚠️ 避免使用 Full Fine-tuning，容易过拟合

### 通用建议

1. **数据质量评估**：在选择 Fine-tuning 方法前，先评估新数据的质量和分布
2. **验证集监控**：密切关注验证集和测试集的表现差异，防止过拟合
3. **方法选择**：
   - 数据质量高 → Full Fine-tuning
   - 数据有噪声/偏移 → Partial Fine-tuning
   - 资源受限/快速实验 → Adapter

---

## 附录：实验配置

所有实验使用相同的超参数配置：
- 基础模型：Transformer (d_model=240, nhead=4, num_layers=4)
- 优化器：Adam (learning_rate=0.00168)
- 训练轮数：最多 150 epochs (early stopping patience=15)
- 批大小：65

输入特征：10个 (TRC-DT, pH-DT, cond-DT, TRC-RT, pH-RT, fDOM-RT, DO-RT, TOC-RT, DOC-RT)
输出目标：10个 (TRC-PPL1/2, pH-PPL1/2, cond-PPL1/2, TOC-PPL1/2, DOC-PPL1/2)

---

*报告生成时间：2026年4月20日*
