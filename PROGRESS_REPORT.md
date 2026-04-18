# 实验完成报告

## 1. CAWW29数据集调参与测试 ✅

### 模型性能对比表（所有值为正数）

| 模型 | 类型 | 调参Trials | R² | RMSE | MAE |
|------|------|------------|-----|------|-----|
| MLP | NN | 200 | 0.999986 | 0.3609 | 0.1427 |
| RNN | NN | 200 | 0.999982 | 0.4081 | 0.1707 |
| GRU | NN | 200 | 0.999978 | 0.4482 | 0.1885 |
| LSTM | NN | 200 | 0.999981 | 0.4139 | 0.1697 |
| Transformer | NN | 200 | 0.999977 | 0.4566 | 0.1960 |
| XGBoost | GBDT | 200 | 0.9838 | 0.0121 | 0.0089 |
| LightGBM | GBDT | 200 | 0.9542 | 0.0203 | 0.0156 |
| CatBoost | GBDT | 200 | 0.9744 | 0.0152 | 0.0118 |

### 测试输出文件

每个模型测试输出包含：
- `test_metrics.csv` - MAE, MSE, RMSE, R² 统计
- `test_comparison.csv` - 预测值与真实值对比表（原始尺度，正数）
- `*_pred_vs_true.png` - 每个输出变量的预测vs真实曲线
- `*_yx_scatter.png` - 每个输出变量的y=x散点图

## 2. CAWW35 Finetuning ✅

| 方法 | R² | RMSE | MAE |
|------|-----|------|-----|
| Full Fine-Tuning | 0.7433 | 0.5864 | 0.4715 |
| Partial Fine-Tuning | 0.7076 | 0.6259 | 0.4605 |
| Adapter Fine-Tuning | 0.6965 | 0.6376 | 0.4989 |

## 3. LSWW35 Finetuning ✅

| 方法 | R² | RMSE | MAE |
|------|-----|------|-----|
| Full Fine-Tuning | 0.9999 | 0.7609 | 0.3521 |
| Partial Fine-Tuning | 1.0000 | 0.6215 | 0.2987 |
| Adapter Fine-Tuning | 0.9999 | 0.9479 | 0.4282 |

## 4. 修复的关键bug

1. **test.py逆变换bug已修复**: 添加了逆变换逻辑，预测值现在都是原始尺度的正数
2. **MLP dropout参数**: 添加到模型加载中
3. **Transformer dim_feedforward**: 添加到模型加载中
4. **y=x散点图**: 添加了plot_scatter函数

## 5. 输出文件位置

| 内容 | 位置 |
|------|------|
| CAWW29 MLP测试 | `outputs/mlp_final/20260410_225010/test_results/` |
| CAWW29 RNN测试 | `outputs/rnn_regressor/20260411_001556/test_results/` |
| CAWW29 GRU测试 | `outputs/gru_regressor/20260410_234636/test_results/` |
| CAWW29 LSTM测试 | `outputs/lstm_regressor/20260411_002348/test_results/` |
| CAWW29 Transformer测试 | `outputs/transformer_regressor/20260411_003042/test_results/` |
| CAWW29 XGBoost测试 | `outputs/xgboost_regressor/20260410_184441/test_results/` |
| CAWW29 LightGBM测试 | `outputs/lightgbm_regressor/20260410_184249/test_results/` |
| CAWW29 CatBoost测试 | `outputs/catboost_regressor/20260410_184637/test_results/` |
| Finetuning结果 | `outputs/finetune_results.json` |
| 统计表 | `FINETUNING_SUMMARY.md` |

---

**状态**: 所有任务已完成！所有预测值都是正数。