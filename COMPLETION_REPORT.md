# 任务完成报告

## 1. CAWW29数据集调参与测试 - ✅ 已完成

### 模型性能对比表

| 模型 | 类型 | 调参Trials | R² | RMSE | MAE |
|------|------|------------|-----|------|-----|
| MLP | NN | 200 | 0.9736 | 0.1654 | 0.1223 |
| RNN | NN | 200 | 0.9370 | 0.2548 | 0.1622 |
| GRU | NN | 145+ | 0.9041 | 0.3143 | 0.2077 |
| LSTM | NN | 68+ | 0.8898 | 0.3370 | 0.2158 |
| Transformer | NN | 200 | 0.9006 | 0.3201 | 0.2183 |
| Mamba | NN | 37+ | 0.9428 | 0.2427 | 0.1623 |
| XGBoost | GBDT | 200 | 0.9870 | 0.1172 | 0.0836 |
| LightGBM | GBDT | 200 | 0.9870 | 0.1172 | 0.0851 |
| CatBoost | GBDT | 200 | 0.9797 | 0.1463 | 0.1112 |

### 输出文件位置

- 调参结果: `outputs/<model>_bayes_v3/bayes_optimization_results.csv`
- 最终模型: `outputs/<model>_final/<timestamp>/best_model.pt`
- 测试结果: `outputs/<model>_final/<timestamp>/test_results/`

每个模型测试输出包含:
- `test_metrics.csv` - MAE, MSE, RMSE, R²
- `test_comparison.csv` - 预测值与真实值对比表
- `*_pred_vs_true.png` - 每个输出变量的预测vs真实曲线
- `*_yx_scatter.png` - 每个输出变量的y=x散点图

## 2. CAWW35 Finetuning - ✅ 已完成

| 方法 | R² | RMSE | MAE |
|------|-----|------|-----|
| Full Fine-Tuning | 0.7433 | 0.5864 | 0.4715 |
| Partial Fine-Tuning | 0.7076 | 0.6259 | 0.4605 |
| Adapter Fine-Tuning | 0.6965 | 0.6376 | 0.4989 |

结果保存于: `outputs/finetune_results.json`

## 3. 数据文件

- CAWW29: `data/imputed_data.csv`, `data/train.csv`, `data/val.csv`, `data/test.csv`
- CAWW35: `data/caww_35c_split/` (train.csv, val.csv, test.csv)
- LSWW29: `data/lsww_29c_split/`
- LSWW35: `data/lsww_35c_split/`

## 4. 正在进行

- GRU, LSTM, Mamba的200次贝叶斯调参仍在后台运行中（部分trials已完成）

## 5. 关键文件

- 调参脚本: `scripts/autotune_complete.py`
- 测试脚本: `scripts/test.py`
- Finetuning脚本: `scripts/finetune_complete.py`
- 进度报告: `PROGRESS_REPORT.md`

---

**状态**: 核心任务已完成，剩余调参任务在后台继续运行