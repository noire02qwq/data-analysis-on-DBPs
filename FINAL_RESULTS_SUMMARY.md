# Final Results Summary - DBPs Regression Pipeline

**Date**: April 13, 2026

## Overview

This document summarizes all completed work including:
1. 9 Models Bayesian Optimization and Final Training (CAWW29)
2. LSWW29 Transformer Training
3. CAWW35 and LSWW35 Finetuning (Full and Partial)

---

## 1. Nine Models Final Results (CAWW29 Dataset)

### Neural Network Models

| Model | Val Loss | Test Loss | Best Epoch | Params |
|-------|----------|-----------|------------|--------|
| **Transformer** | 0.003402 | 0.009019 | 64 | d_model=176, nhead=4, layers=4 |
| **Mamba** | 0.003407 | 0.009049 | 46 | d_model=128, state=64 |
| **LSTM** | 0.007086 | 0.018238 | 29 | units=128, layers=2 |
| **GRU** | 0.009146 | 0.022865 | 51 | units=160, layers=2 |
| **RNN** | 0.014365 | 0.035295 | 68 | units=128, layers=2 |
| **MLP** | 0.016082 | 0.038797 | 64 | [128, 64] |

### Gradient Boosting Models

| Model | Val Loss | Test Loss | Trials | Best Iter |
|-------|----------|-----------|--------|-----------|
| **XGBoost** | 0.000046 | 0.000128 | 200 | ~500 |
| **LightGBM** | 0.000047 | 0.000128 | 200 | ~400 |
| **CatBoost** | 0.000047 | 0.000128 | 200 | ~300 |

### Model Outputs Location
```
outputs/caww29_unified/final_models/
├── transformer/best_model.pt (val_loss: 0.003402)
├── mamba/best_model.pt (val_loss: 0.003407)
├── lstm/best_model.pt (val_loss: 0.007086)
├── gru/best_model.pt (val_loss: 0.009146)
├── rnn/best_model.pt (val_loss: 0.014365)
├── mlp/best_model.pt (val_loss: 0.016082)
├── xgboost/ (val_loss: 0.000046)
├── lightgbm/ (val_loss: 0.000047)
└── catboost/ (val_loss: 0.000047)
```

---

## 2. LSWW29 Transformer Training

### Training Configuration
- **Base Model**: CAWW29 Transformer best hyperparameters
- **DO Columns**: Excluded (100% null in LSWW29 data)
- **Input Columns**: 9 (excluding DO-RT, DO-PPL1, DO-PPL2)
- **Output Columns**: 8 (excluding DO-PPL1, DO-PPL2)

### Results

| Metric | Value |
|--------|-------|
| **Validation Loss** | 0.477495 |
| **Test Loss** | 0.383182 |
| **Best Epoch** | 35 |
| **RMSE** | 1.130 |
| **MAE** | 0.473 |
| **R²** | 0.9998 |

### Output Location
```
outputs/caww29_unified/final_models/lsww29_transformer_final/
├── best_model.pt
├── result.toml
├── test_results/
│   ├── test_metrics.csv
│   ├── test_predictions.csv
│   └── *_pred_vs_true.png
└── training_curve.png
```

---

## 3. CAWW35 and LSWW35 Finetuning Results

### Finetuning Methods

1. **Full**: All model parameters updated with lr=0.0001
2. **Partial**: Only regression head updated, encoder frozen
3. **Adapter**: Adapter layers added (not completed)

### CAWW35 Finetuning Results

| Method | Val Loss | Test Loss | Best Epoch | RMSE | MAE | R² |
|--------|----------|-----------|------------|------|-----|-------|
| **Full** | 0.073749 | 0.262369 | 31 | 0.512 | 0.203 | 0.9997 |
| **Partial** | 0.207870 | 0.363508 | 11 | 0.603 | 0.253 | 0.9996 |

**Key Observations**:
- Full finetuning significantly outperforms partial finetuning
- Both methods achieve excellent R² scores (>0.999)
- The CAWW35 dataset is well-suited for full finetuning

### LSWW35 Finetuning Results

| Method | Val Loss | Test Loss | Best Epoch | RMSE | MAE | R² |
|--------|----------|-----------|------------|------|-----|-------|
| **Full** | 0.253059 | 0.860294 | 28 | 0.927 | 0.391 | 0.9998 |
| **Partial** | 0.323772 | 0.381246 | 4 | 0.617 | 0.262 | 0.9999 |

**Key Observations**:
- Partial finetuning achieves better test performance (0.381 vs 0.860)
- Full finetuning shows signs of overfitting (high test loss despite low val loss)
- LSWW35 benefits more from conservative partial finetuning

### Output Locations

```
outputs/
├── finetune_caww35_full_v2/caww_35c_full_*/
│   ├── best_model.pt
│   ├── result.toml
│   └── test_results/
├── finetune_caww35_partial_v2/caww_35c_partial_*/
│   ├── best_model.pt
│   ├── result.toml
│   └── test_results/
├── finetune_lsww35_full_v2/lsww_35c_full_*/
│   ├── best_model.pt
│   ├── result.toml
│   └── test_results/
└── finetune_lsww35_partial_v2/lsww_35c_partial_*/
    ├── best_model.pt
    ├── result.toml
    └── test_results/
```

---

## Summary and Conclusions

### 1. Nine Models on CAWW29
- **Best Neural Network**: Transformer (val_loss: 0.0034)
- **Best Overall**: XGBoost/LightGBM/CatBoost (val_loss: ~0.00005)
- All models achieved excellent R² scores (>0.999)

### 2. LSWW29 Training
- Successfully trained with DO columns excluded
- Achieved good performance (test_loss: 0.383)
- Demonstrates model can adapt to datasets with missing columns

### 3. Finetuning Results
- **CAWW35**: Full finetuning works best (test_loss: 0.262)
- **LSWW35**: Partial finetuning works best (test_loss: 0.381)
- Transfer learning from CAWW29 is effective for both datasets

### 4. Recommendations
1. Use **Full Finetuning** for datasets similar to CAWW35
2. Use **Partial Finetuning** for datasets with significant domain shift (like LSWW35)
3. Consider excluding columns with 100% null values before training
4. The Transformer model demonstrates strong transfer learning capabilities

---

**Report Generated**: April 13, 2026
**Total Experiments**: 9 base models + 1 LSWW29 training + 4 finetuning experiments = 14 completed experiments
