# Finetune 结果汇总报告

**生成时间**: 2026-04-12  
**生成命令**: transfer_learning.py

---

## 执行概要

所有finetune任务已使用`transfer_learning.py`脚本执行完成。

### 执行命令

```bash
# CAWW35 Finetune (3种模式)
python scripts/transfer_learning.py --task caww35 --mode full
python scripts/transfer_learning.py --task caww35 --mode partial
python scripts/transfer_learning.py --task caww35 --mode frozen

# LSWW35 Finetune (3种模式)
python scripts/transfer_learning.py --task lsw35 --mode full
python scripts/transfer_learning.py --task lsw35 --mode partial
python scripts/transfer_learning.py --task lsw35 --mode frozen
```

---

## CAWW35 Finetune 结果

| 模式 | 状态 | 说明 |
|------|------|------|
| Full | 完成 | 全部参数可训练 |
| Partial | 完成 | 冻结部分层 |
| Frozen | 完成 | 冻结特征提取器 |

**输出目录**:
- `caww35_full/`
- `caww35_partial/`
- `caww35_frozen/`

---

## LSWW35 Finetune 结果

| 模式 | 状态 | 说明 |
|------|------|------|
| Full | 完成 | 全部参数可训练 |
| Partial | 完成 | 冻结部分层 |
| Frozen | 完成 | 冻结特征提取器 |

**输出目录**:
- `lsw35_full/`
- `lsw35_partial/`
- `lsw35_frozen/`

---

## Finetune 模式说明

### Full (全参数微调)
- 所有模型参数均可训练
- 学习率: 较小 (通常0.0001)
- 适用: 目标数据集与源数据集差异较大时

### Partial (部分层冻结)
- 冻结底层特征提取层
- 只训练上层分类/回归层
- 学习率: 中等 (通常0.0005)
- 适用: 目标数据集与源数据集有一定相似性

### Frozen (冻结特征提取器)
- 完全冻结预训练模型的特征提取部分
- 只训练最后的回归头
- 学习率: 较大 (通常0.001)
- 适用: 目标数据集较小，需要防止过拟合

---

## 总结

- **CAWW35 Finetune**: 3种模式全部完成
- **LSWW35 Finetune**: 3种模式全部完成
- **总计**: 6个finetune任务全部完成

---

**报告结束**
