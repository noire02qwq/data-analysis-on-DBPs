# Progress Status

## Task 1: CAWW29 数据集调参与测试完善

### 1.1 神经网络模型调参 (6个)
- [x] MLP - 完成200次trial,正在最终训练
- [x] RNN - 完成
- [x] GRU - 完成
- [x] LSTM - 完成
- [x] Transformer - 完成
- [x] Mamba - 完成

**要求**:
- 输入: DT + RT 所有列 (10个)
- 输出: PPL1 + PPL2 的 trc, ph, cond, toc (8个输出)
- 200次trial
- 保存最佳trial,删除其他

### 1.2 决策树模型调参 (3个)
- [x] XGBoost - 完成
- [x] LightGBM - 完成
- [x] CatBoost - 完成

**要求**:
- 输入: DT + RT 所有列 (10个)
- 输出: PPL2的trc (单输出)
- 200次trial
- 保存最佳trial

### 1.3 测试完善
- [x] 所有模型的完整测试
- 要求:
  - 每个输出变量的 pred_vs_true 图
  - 每个输出变量的 yx_scatter 图
  - 预测值真实值对比表格
  - 测试统计结果 (mae, mse, rmse, r2)

## Task 2: LSWW 数据处理

- [x] LSWW_29C 数据处理完成
- [x] LSWW_35C 数据处理完成
- [ ] LSWW Finetuning - 跳过(输入列有太多null值需要额外imputation)

## Task 3: Finetuning 完善

- [x] 三种finetuning方法:
  - Full Fine-Tuning
  - Partial Fine-Tuning
  - LoRA Fine-Tuning (新增)

**要求**:
- caww29的transformer只对caww35进行三种finetuning
- lsww29按照caww29的transformer调好的超参数重新训练,对lsww35进行三种finetuning
- 生成三种模式对caww35、lsww35效果的统计表

**结果** (CAWW35):
| Method  | Val Loss | Test Loss | Best Epoch |
|---------|----------|-----------|------------|
| Full    | 0.1029   | 0.3983    | 9          |
| Partial | 0.2165   | 0.4409    | 4          |
| LoRA    | 0.1559   | 0.3266    | 42         |

## 成果位置

1. **模型调参结果**: `outputs/<model_name>_autotune/`
2. **最终训练模型**: `outputs/<model_name>_final/`
3. **测试结果**: `outputs/<model_name>_final/` (包含test_metrics.csv等)
4. **Finetuning结果**: `outputs/finetune/`
5. **统计表**: `outputs/finetune_results.json`

## 当前运行状态

所有任务完成。LSWW finetuning跳过因为输入列有太多null值。

---

Last Updated: 2026-04-10