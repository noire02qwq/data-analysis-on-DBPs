# DBPs Temporal Experiment - Complete Workflow Status

## Last Updated: 2026-04-11

---

## 1. DATA PIPELINE (VERIFIED ✓)

### Temporal Split (NO SHUFFLING)
```
Train: 2025/5/17 1:50 → 2025/6/14 9:10 (8142 samples) - EARLIEST
Val:   2025/6/14 9:15 → 2025/6/20 10:30 (1744 samples) - MIDDLE
Test:  2025/6/20 10:35 → 2025/6/26 12:00 (1746 samples) - LATEST
```

### Column Configuration
- **Input columns**: TRC-DT, pH-DT, cond-DT, TRC-RT, pH-RT, cond-RT, fDOM-RT, DO-RT, TOC-RT, DOC-RT
- **Output columns**: TRC-PPL1, TRC-PPL2, pH-PPL1, pH-PPL2, cond-PPL1, cond-PPL2, TOC-PPL1, TOC-PPL2

---

## 2. EXPERIMENT CONFIGURATION

### Models (9 Total)
| # | Model | Type | Status |
|---|-------|------|--------|
| 1 | XGBoost | GBDT | Running (95/200 trials) |
| 2 | LightGBM | GBDT | Pending |
| 3 | CatBoost | GBDT | Pending |
| 4 | MLP | NN | Pending |
| 5 | RNN | NN | Pending |
| 6 | GRU | NN | Pending |
| 7 | LSTM | NN | Pending |
| 8 | Mamba | NN | Pending |
| 9 | Transformer | NN | Pending |

### Bayesian Optimization Settings
- **Trials per model**: 200
- **Seed**: 42
- **Timeout per trial**: 600 seconds (10 minutes)
- **Optimization metric**: Validation loss (MSE)

---

## 3. OUTPUT STRUCTURE

### Directory Layout
```
outputs/temporal_experiment/
├── experiment_summary.json          # Overall results
└── models/
    └── {model_key}/                 # e.g., xgboost, lightgbm, etc.
        ├── bayes_opt/
        │   ├── trials/
        │   │   └── trial_{N:03d}_config.toml
        │   └── {model}_bayes_results.csv
        ├── best_config.toml
        ├── final_model/
        │   ├── best_model.pt
        │   ├── config.toml
        │   ├── result.toml
        │   ├── scalers.npz
        │   ├── loss_history.csv
        │   └── training_curve.png
        └── test_results/
            ├── test_metrics.csv
            ├── test_predictions.csv
            ├── test_comparison.csv
            ├── {target}_pred_vs_true.png  # With day-level timestamps
            └── {target}_yx_scatter.png
```

---

## 4. VISUALIZATION SPECIFICATIONS

### Prediction Plots (`{target}_pred_vs_true.png`)
- **X-axis**: Date (day-level, format: `%Y/%m/%d`)
- **Locator**: Every 2 days (`DayLocator(interval=2)`)
- **Rotation**: 45 degrees
- **Y-axis**: True and Predicted values
- **Content**: Line plot with True (blue) and Predicted (red) values

### Scatter Plots (`{target}_yx_scatter.png`)
- **X-axis**: True values
- **Y-axis**: Predicted values
- **Content**: Scatter plot with y=x reference line
- **Metrics**: R² and RMSE displayed

---

## 5. FINE-TUNING TASKS (After CAWW29 completes)

### Task 1: LSWW29 New Training
- **Base**: None (train from scratch)
- **Data**: LSWW29 dataset
- **Hyperparameters**: Use best from CAWW29 Transformer
- **Output**: `outputs/transfer_learning/lsw29_new/`

### Task 2: CAWW35 Fine-tuning (3 modes)
- **Base**: CAWW29 trained model
- **Data**: CAWW35 dataset
- **Modes**:
  1. Full fine-tuning (all parameters)
  2. Partial fine-tuning (freeze encoder)
  3. Frozen feature extractor
- **Output**: `outputs/transfer_learning/caww35_{mode}/`

### Task 3: LSWW35 Fine-tuning (3 modes)
- **Base**: LSWW29 trained model
- **Data**: LSWW35 dataset
- **Modes**: Same as CAWW35
- **Output**: `outputs/transfer_learning/lsww35_{mode}/`

---

## 6. COMPLETION CHECKLIST

### Phase 1: CAWW29 Temporal Experiment
- [ ] XGBoost (200 trials) - In Progress (~48%)
- [ ] LightGBM (200 trials) - Pending
- [ ] CatBoost (200 trials) - Pending
- [ ] MLP (200 trials) - Pending
- [ ] RNN (200 trials) - Pending
- [ ] GRU (200 trials) - Pending
- [ ] LSTM (200 trials) - Pending
- [ ] Mamba (200 trials) - Pending
- [ ] Transformer (200 trials) - Pending

### Phase 2: Fine-tuning Tasks
- [ ] LSWW29 New Training
- [ ] CAWW35 Fine-tuning (3 modes)
- [ ] LSWW35 Fine-tuning (3 modes)

### Documentation
- [ ] Update RESULTS_SUMMARY_FINAL.md
- [ ] Document temporal split methodology
- [ ] Record all hyperparameters and results

---

## 7. CURRENT STATUS

**Experiment Started**: 2026-04-11 19:52:56
**Current Time**: $(date)
**Active PID**: 1125462
**Current Model**: XGBoost (38/200 trials, ~19%)
**Estimated Time Remaining**: 
- XGBoost: ~8.5 minutes remaining
- All 9 models: ~90 minutes total

---

**Next Update**: Check progress in 10 minutes
