# CAWW29 数据集实验完成总结

## 修复的问题
1. **test.py逆变换bug已修复**: 添加了dropout参数读取和逆变换逻辑
   - 之前：预测值是标准化后的值（包含负数）
   - 现在：正确逆变换到原始尺度（所有值为正数）

## 模型测试结果

### 神经网络模型 (多输出: TRC, pH, cond, TOC → PPL1, PPL2)

| 模型 | Test R² | Test RMSE | Test MAE | 模型路径 |
|------|---------|-----------|----------|----------|
| MLP | 0.999986 | 0.3609 | 0.1427 | `outputs/mlp_final/20260410_225010/` |
| RNN | 0.999982 | 0.4081 | 0.1707 | `outputs/rnn_regressor/20260411_001556/` |
| GRU | 0.999978 | 0.4482 | 0.1885 | `outputs/gru_regressor/20260410_234636/` |
| LSTM | 0.999981 | 0.4139 | 0.1697 | `outputs/lstm_regressor/20260411_002348/` |
| Transformer | 0.999977 | 0.4566 | 0.1960 | `outputs/transformer_regressor/20260411_003042/` |
| Mamba | 0.999976 | - | - | `outputs/mamba_regressor/20260411_003644/` |

### 决策树模型 (单输出: TRC-PPL2)

| 模型 | Test R² | Test RMSE | Test MAE | 模型路径 |
|------|---------|-----------|----------|----------|
| XGBoost | 0.983839 | 0.0121 | 0.0089 | `outputs/xgboost_regressor/20260410_184441/` |
| LightGBM | 0.954176 | 0.0203 | 0.0156 | `outputs/lightgbm_regressor/20260410_184249/` |
| CatBoost | 0.974368 | 0.0152 | 0.0118 | `outputs/catboost_regressor/20260410_184637/` |

## 测试输出文件

每个模型测试输出包含（都在 `test_results/` 目录下）:
- `test_metrics.csv` - MAE, MSE, RMSE, R² 统计
- `test_comparison.csv` - 预测值与真实值对比表（原始尺度，正数值）
- `*_pred_vs_true.png` - 每个输出变量的预测vs真实曲线
- `*_yx_scatter.png` - 每个输出变量的y=x散点图

## 调参结果文件

- 调参结果CSV: `outputs/<model>_bayes_v3/bayes_optimization_results.csv`
- 最佳trial: `outputs/<model>_bayes_v3/trial_*/`

## 关键修复

test.py 中修复了两个bug:
1. MLP模型未读取dropout参数，导致模型结构不匹配
2. Transformer模型未读取dim_feedforward参数
3. 所有模型添加了逆变换逻辑，将标准化后的预测值转换回原始尺度

---

**注意**: 预测值现在都是正数（化学浓度、pH、导电率、TOC等），符合物理意义。