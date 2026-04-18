# Final Complete Results Summary

**Date**: April 13, 2026
**Status**: All Tasks Completed ✓

---

## 1. LSWW29 Transformer Training (完成)

Using original CAWW29 Transformer hyperparameters with DO columns excluded.

| Metric | Value |
|--------|-------|
| **Val Loss** | 0.477495 |
| **Test Loss** | 0.383182 |
| **Best Epoch** | 35 |
| **RMSE** | 1.130 |
| **MAE** | 0.473 |
| **R²** | 0.9998 |

**Output Location**: `outputs/caww29_unified/final_models/lsww29_transformer_final/`

---

## 2. CAWW35 Finetuning Results (三种模式全部完成)

### 2.1 Full Finetuning

| Metric | Value |
|--------|-------|
| **Val Loss** | 0.073749 |
| **Test Loss** | 0.262369 |
| **Best Epoch** | 31 |

**Location**: `outputs/finetune_caww35_full_v2/`

### 2.2 Partial Finetuning

| Metric | Value |
|--------|-------|
| **Val Loss** | 0.207870 |
| **Test Loss** | 0.363508 |
| **Best Epoch** | 11 |

**Location**: `outputs/finetune_caww35_partial_v2/`

### 2.3 Adapter Finetuning ✓ (Completed)

| Metric | Value |
|--------|-------|
| **Val Loss** | 0.357208 |
| **Test Loss** | 0.645785 |
| **Best Epoch** | 45 |
| **Trainable Params** | 72,334 / 1,128,336 (6.41%) |

**Location**: `outputs/finetune_caww35_adapter_v3/`

---

## 3. LSWW35 Finetuning Results (三种模式全部完成)

### 3.1 Full Finetuning

| Metric | Value |
|--------|-------|
| **Val Loss** | 0.253059 |
| **Test Loss** | 0.860294 |
| **Best Epoch** | 28 |

**Location**: `outputs/finetune_lsww35_full_v2/`

### 3.2 Partial Finetuning

| Metric | Value |
|--------|-------|
| **Val Loss** | 0.323772 |
| **Test Loss** | 0.381246 |
| **Best Epoch** | 4 |

**Location**: `outputs/finetune_lsww35_partial_v2/`

### 3.3 Adapter Finetuning ✓ (Completed)

| Metric | Value |
|--------|-------|
| **Val Loss** | 0.318619 |
| **Test Loss** | 0.458358 |
| **Best Epoch** | 1 |
| **Trainable Params** | 71,050 / 1,127,052 (6.30%) |

**Location**: `outputs/finetune_lsww35_adapter_v3/`

---

## 4. 9 Models Final Results (CAWW29 Dataset)

All models completed with 200-trial Bayesian optimization.

| Model | Val Loss | Test Loss | Best Epoch |
|-------|----------|-----------|------------|
| **Transformer** | 0.003402 | 0.009019 | 64 |
| **Mamba** | 0.003407 | 0.009049 | 46 |
| **LSTM** | 0.007086 | 0.018238 | 29 |
| **GRU** | 0.009146 | 0.022865 | 51 |
| **RNN** | 0.014365 | 0.035295 | 68 |
| **MLP** | 0.016082 | 0.038797 | 64 |
| **XGBoost** | 0.000046 | 0.000128 | N/A |
| **LightGBM** | 0.000047 | 0.000128 | N/A |
| **CatBoost** | 0.000047 | 0.000128 | N/A |

**Location**: `outputs/caww29_unified/final_models/`

---

## Summary of All Results

### Completed Experiments: 15 Total

| # | Experiment | Status | Location |
|---|------------|--------|----------|
| 1 | LSWW29 Training | ✓ | `outputs/caww29_unified/final_models/lsww29_transformer_final/` |
| 2 | CAWW35 Full | ✓ | `outputs/finetune_caww35_full_v2/` |
| 3 | CAWW35 Partial | ✓ | `outputs/finetune_caww35_partial_v2/` |
| 4 | CAWW35 Adapter | ✓ | `outputs/finetune_caww35_adapter_v3/` |
| 5 | LSWW35 Full | ✓ | `outputs/finetune_lsww35_full_v2/` |
| 6 | LSWW35 Partial | ✓ | `outputs/finetune_lsww35_partial_v2/` |
| 7 | LSWW35 Adapter | ✓ | `outputs/finetune_lsww35_adapter_v3/` |
| 8-15 | 9 Models CAWW29 | ✓ | `outputs/caww29_unified/final_models/` |

---

## Documentation Files

- `FINAL_COMPLETE_SUMMARY.md` (this file)
- `FINAL_RESULTS_SUMMARY.md`
- `FINETUNING_RESULTS_SUMMARY.md`
- `CLAUDE.md`

---

**All tasks completed successfully!** ✓
