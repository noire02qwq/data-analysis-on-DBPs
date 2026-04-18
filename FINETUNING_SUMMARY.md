# Finetuning 实验结果统计表

## CAWW35 Finetuning (从CAWW29 Transformer)

| 方法 | R² | RMSE | MAE |
|------|-----|------|-----|
| Full Fine-Tuning | 0.7433 | 0.5864 | 0.4715 |
| Partial Fine-Tuning | 0.7076 | 0.6259 | 0.4605 |
| Adapter Fine-Tuning | 0.6965 | 0.6376 | 0.4989 |

## LSWW35 Finetuning (从LSWW29 Transformer)

| 方法 | R² | RMSE | MAE |
|------|-----|------|-----|
| Full Fine-Tuning | 0.9999 | 0.7609 | 0.3521 |
| Partial Fine-Tuning | 1.0000 | 0.6215 | 0.2987 |
| Adapter Fine-Tuning | 0.9999 | 0.9479 | 0.4282 |

## 总结

- **CAWW35**: Full Fine-Tuning效果最好 (R²=0.7433)
- **LSWW35**: Partial Fine-Tuning效果最好 (R²≈1.0)

## 文件位置

- CAWW35结果: `outputs/finetune_results.json` (caww35_full, caww35_partial, caww35_lora)
- LSWW35结果: `outputs/finetune_results.json` (lsww35_full, lsww35_partial, lsww35_adapter)
- LSWW29训练模型: `outputs/lsww29_transformer/`

## Finetuning方法说明

1. **Full Fine-Tuning**: 更新所有模型参数
2. **Partial Fine-Tuning**: 冻结encoder，只训练head
3. **Adapter Fine-Tuning**: 冻结大部分层，只训练head和input_projection