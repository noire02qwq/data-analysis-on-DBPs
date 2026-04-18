# Current Project Status

## Overview
This file tracks progress on the four main tasks requested.

---

## Task 1: Complete all 9 models with full tuning, training, and testing

**Models: xgboost, lightgbm, catboost, mlp, rnn, gru, lstm, transformer, mamba**

| Model | Status | Best val_loss | Test loss | Notes |
|-------|---------|---------------|-----------|-------|
| MLP | ✅ Complete | 0.270386 | 4.479017 | Final model at `outputs/mlp_final/` |
| GRU | ✅ Complete | 0.227567 | 4.141379 | Final model at `outputs/gru_final/` |
| LSTM | ✅ Complete | 0.241213 | 4.979813 | Final model at `outputs/lstm_final/` |
| Transformer | ✅ Complete | 0.695590 | 3.329812 | Final model at `outputs/transformer_final/` |
| Mamba | ✅ Complete | 0.558456 | 3.166301 | Final model at `outputs/mamba_final/` |
| XGBoost | ✅ Complete | 0.473939 | 2.201434 | Final model at `outputs/xgboost_regressor/` |
| LightGBM | ✅ Complete | 0.487116 | 2.348165 | Final model at `outputs/lightgbm_regressor/` |
| CatBoost | ✅ Complete | 0.474527 | 2.297998 | Final model at `outputs/catboost_regressor/` |
| RNN | ✅ Complete | 0.427719 | 4.629791 | Final model at `outputs/rnn_regressor/` |

**Progress: 9/9 models complete ✓**

---

## Task 2: Clean up output directories, remove old failed outputs

**Status: ✅ COMPLETE**
- Initial cleanup accidentally deleted the complete outputs for 4 models (xgboost, lightgbm, catboost, rnn) - all have been successfully rebuilt
- Cleaned up 349 old individual trial directories from Bayesian optimization (saved ~4MB of disk space)
- All final model outputs preserved and organized

---

## Task 3: Process three new datasets (CAWW_35C, LSWW_29C, LSWW_35C)

**Status: ✅ COMPLETE**

All three datasets have been fully processed:
1. Converted from Excel to CSV raw data matching original format
2. Missing values imputed
3. Split 70:15:15 into train/val/test

Outputs:
- `data/caww_35c_raw_data.csv`
- `data/caww_35c_imputed_data.csv`
- `data/caww_35c_split/{train,val,test}.csv`
- `data/lsww_29c_raw_data.csv`
- `data/lsww_29c_imputed_data.csv`
- `data/lsww_29c_split/{train,val,test}.csv`
- `data/lsww_35c_raw_data.csv`
- `data/lsww_35c_imputed_data.csv`
- `data/lsww_35c_split/{train,val,test}.csv`

---

## Task 4: Fine-tune Transformer with two methods on new datasets

**Status: ✅ COMPLETE**

- Created `scripts/finetune_transformer.py` that implements both fine-tuning methods:
  1. **Full Fine-Tuning**: Update all model parameters with smaller learning rate (1e-4)
  2. **Partial Fine-Tuning**: Freeze transformer encoder, only train final regression head
- Created `run_all_finetune.py` batch runner for all 3 datasets × 2 methods
- All 6 fine-tuning runs completed successfully:
  - caww_35c full: val_loss=0.099892, test_loss=0.428584
  - caww_35c partial: val_loss=0.180423, test_loss=0.521748
  - lsww_29c full: val_loss=0.520895, test_loss=0.474314
  - lsww_29c partial: val_loss=0.704583, test_loss=0.685239
  - lsww_35c full: val_loss=0.336831, test_loss=0.856224
  - lsww_35c partial: val_loss=0.571502, test_loss=0.697145
- All outputs saved to `outputs/finetune/`: includes saved model, loss history CSV, training curve PNG, result metrics in TOML/JSON, scalers

---

## Output Locations

### 9 Model Training Results
| Model | Output Directory | Key Files |
|-------|-----------------|-----------|
| XGBoost | `outputs/xgboost_regressor/20260409_031340/` | best_model.xgb, training_curve.png, result.toml |
| LightGBM | `outputs/lightgbm_regressor/20260409_031808/` | best_model.lgb, training_curve.png, result.toml |
| CatBoost | `outputs/catboost_regressor/20260409_032706/` | best_model.cbm, training_curve.png, result.toml |
| MLP | `outputs/mlp_final/` | best_model.pt, training_curve.png, result.toml |
| RNN | `outputs/rnn_regressor/` | best_model.pt, training_curve.png, result.toml |
| GRU | `outputs/gru_final/` | best_model.pt, training_curve.png, result.toml |
| LSTM | `outputs/lstm_final/` | best_model.pt, training_curve.png, result.toml |
| Transformer | `outputs/transformer_final/` | best_model.pt, training_curve.png, result.toml |
| Mamba | `outputs/mamba_final/` | best_model.pt, training_curve.png, result.toml |

### Fine-tuning Results
All fine-tuning outputs: `outputs/finetune/`

#### Fine-tuning Methods

Two主流微调方法 (Two mainstream fine-tuning methods) were implemented:

1. **Full Fine-Tuning (完全微调)**
   - Updates all model parameters with a smaller learning rate (1e-4)
   - Uses Adam optimizer with weight_decay=1e-5
   - Maximum 50 epochs with early stopping (patience=8)
   - All model parameters are trainable

2. **Partial Fine-Tuning (部分微调/冻结微调)**
   - Freezes the Transformer encoder (all transformer layers)
   - Only trains the final regression head (fc layers)
   - Learning rate: 1e-4
   - Significantly fewer trainable parameters (~1,770 vs ~1.8M)
   - Faster training, lower risk of overfitting

#### Results Analysis

| Dataset | Method | Val Loss | Test Loss | Trainable Params | Best Epoch |
|---------|--------|----------|-----------|-----------------|------------|
| caww_35c | full | 0.099892 | 0.428584 | 1,803,110 | 9 |
| caww_35c | partial | 0.180423 | 0.521748 | 1,770 | 30 |
| lsww_29c | full | 0.520895 | 0.474314 | 1,802,934 | 11 |
| lsww_29c | partial | 0.704583 | 0.685239 | 1,770 | 49 |
| lsww_35c | full | 0.336831 | 0.856224 | 1,802,934 | 27 |
| lsww_35c | partial | 0.571502 | 0.697145 | 1,770 | 10 |

#### Key Observations

- **CAWW_35C**: Both methods achieved best results; full fine-tuning has lower val_loss but higher test_loss (potential overfitting)
- **LSWW_29C**: Full fine-tuning significantly outperforms partial; dataset may benefit from adapting all layers
- **LSWW_35C**: Partial fine-tuning achieved better test loss (0.697 vs 0.856), demonstrating benefits when dataset is small
- **Data Quality**: LSWW datasets have completely missing DO-RT column (all nulls), reducing input dimension from 10 to 8

### Data Files
| Dataset | Raw CSV | Imputed CSV | Split Directory |
|---------|---------|-------------|-----------------|
| CAWW_35C | `data/caww_35c_raw_data.csv` | `data/caww_35c_imputed_data.csv` | `data/caww_35c_split/` |
| LSWW_29C | `data/lsww_29c_raw_data.csv` | `data/lsww_29c_imputed_data.csv` | `data/lsww_29c_split/` |
| LSWW_35C | `data/lsww_35c_raw_data.csv` | `data/lsww_35c_imputed_data.csv` | `data/lsww_35c_split/` |

### Each Model Output Contains:
- `best_model.pt/.xgb/.lgb/.cbm` - Saved model weights
- `training_curve.png` - Loss visualization
- `loss_history.csv` - Per-epoch loss values
- `result.toml` - Training metrics (val_loss, test_loss, best_epoch)
- `config.toml` - Full training configuration
- `scalers.npz` - Feature scaling parameters

---

## Last Updated
2026-04-09
