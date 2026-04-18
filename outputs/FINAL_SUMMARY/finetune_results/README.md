# Transformer Finetune 结果汇总

## 实验设置

基于CAWW29最佳Transformer模型，在CAWW35和LSWW35数据集上进行三种finetune模式：

### Finetune 模式

| 模式 | 冻结层 | 学习率 | 说明 |
|------|--------|--------|------|
| full | 无 | 0.0001 | 全部参数可训练 |
| partial | encoder.layers.0, encoder.layers.1 | 0.0005 | 冻结前2层encoder |
| frozen | encoder | 0.001 | 冻结整个encoder，只训练head |

## 数据集

- **CAWW35**: 35°C CAWW数据集，与CAWW29同类型
- **LSWW35**: 35°C LSWW数据集，不同水质类型

## 结果文件

- `caww35_{full,partial,frozen}/`: CAWW35三种模式结果
- `lsww35_{full,partial,frozen}/`: LSWW35三种模式结果
