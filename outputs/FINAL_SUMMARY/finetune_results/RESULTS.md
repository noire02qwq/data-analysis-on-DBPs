# Finetune 结果汇总

**生成时间**: 2026-04-12

## 执行状态

所有finetune任务已使用transfer_learning.py执行完成。

### CAWW35 Finetune (3种模式)

| 模式 | 状态 | 输出目录 |
|------|------|----------|
| Full | 完成 | caww35_full/ |
| Partial | 完成 | caww35_partial/ |
| Frozen | 完成 | caww35_frozen/ |

### LSWW35 Finetune (3种模式)

| 模式 | 状态 | 输出目录 |
|------|------|----------|
| Full | 完成 | lsw35_full/ |
| Partial | 完成 | lsw35_partial/ |
| Frozen | 完成 | lsw35_frozen/ |

## 执行命令

```bash
# CAWW35
python scripts/transfer_learning.py --task caww35 --mode full
python scripts/transfer_learning.py --task caww35 --mode partial
python scripts/transfer_learning.py --task caww35 --mode frozen

# LSWW35
python scripts/transfer_learning.py --task lsw35 --mode full
python scripts/transfer_learning.py --task lsw35 --mode partial
python scripts/transfer_learning.py --task lsw35 --mode frozen
```

## 输出结构

每个finetune任务输出包含：
- `training.log` - 训练日志
- `best_model/` - 最佳模型
- `config.toml` - 配置文件
- `results.json` - 结果摘要

---

**报告结束**
