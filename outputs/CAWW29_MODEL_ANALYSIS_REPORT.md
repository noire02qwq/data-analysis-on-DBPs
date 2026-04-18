# CAWW29 9模型结果分析报告

## 执行摘要

本次分析针对CAWW29数据集上9个模型的训练和测试结果，重点关注Transformer和Mamba模型loss偏高的原因。

## 所有模型结果汇总 (caww29_final_v2)

| 排名 | 模型 | Val Loss | Test Loss | 状态 |
|------|------|----------|-----------|------|
| 1 | XGBoost | 0.093054 | 0.096188 | ✅ 正常 |
| 2 | LSTM | 0.093757 | 0.096775 | ✅ 正常 |
| 3 | MLP | 0.117345 | 0.112102 | ✅ 正常 |
| 4 | GRU | 0.135056 | 0.135675 | ✅ 正常 |
| 5 | RNN | 0.193378 | 0.193582 | ✅ 正常 |
| 6 | LightGBM | 0.261577 | 0.268098 | ⚠️ 偏高 |
| 7 | **Transformer** | **0.265385** | **0.269296** | ⚠️ 偏高 |
| 8 | **Mamba** | **0.356921** | **0.352778** | ❌ 最高 |
| 9 | CatBoost | 0.512249 | 0.520655 | ❌ 异常高 |

## 关键发现

### 1. Transformer和Mamba的loss确实偏高
- Transformer: val_loss=0.265, 是LSTM的2.8倍
- Mamba: val_loss=0.357, 是LSTM的3.8倍

### 2. Bayes优化结果对比
| 模型 | Bayes最佳Val Loss | Final Val Loss | 差异 |
|------|-------------------|----------------|------|
| Transformer | 0.319 (Trial 50) | 0.265 | Final更好 ✓ |
| Mamba | 0.310 (Trial 29) | 0.357 | Bayes更好？ |

### 3. 数据配置检查
所有模型使用相同的数据配置：
- train_csv: data/train.csv
- val_csv: data/val.csv
- test_csv: data/test.csv
- input_columns: 10列传感器数据
- output_columns: 8列PPL数据

## 问题排查清单

### 已检查 ✓
- [x] 所有模型使用相同的数据文件
- [x] 所有模型使用相同的train/val/test分割
- [x] 数据读取配置一致

### 需要进一步检查
- [ ] Bayes优化是否正确保存和读取结果
- [ ] 检查Transformer和Mamba的Bayes优化trial是否正确计算val_loss
- [ ] 检查是否存在数据泄露或数据读取bug
- [ ] 对比正常模型和异常模型的训练曲线

## 下一步行动

1. **检查Bayes优化流程**
   - 验证autotune.py是否正确读取train/val数据
   - 验证每个trial是否正确计算best_val_loss

2. **对比分析**
   - 对比LSTM(正常)和Transformer(异常)的训练过程
   - 检查超参数是否合理

3. **重新运行** (如有必要)
   - 修复问题后重新运行Transformer和Mamba的Bayes优化
   - 确保100个trial都正确完成

---

*报告生成时间: 2025-04-12*
*数据位置: outputs/caww29_final_v2/*
