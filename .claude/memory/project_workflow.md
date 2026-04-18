---
name: DBPs ML Pipeline Workflow
description: Completed neural network and GBDT workflows for DBPs regression
type: project
---

# Completed Work - April 6, 2026

## 1. CatBoost GPU Experiment (Fixed)
**Script**: `run_catboost_gpu_only.py`
**Status**: Running (66/100 trials completed)
**Key Fix**: Used `bootstrap_type: 'Poisson'` to support `subsample < 1.0` with GPU

**Outputs**:
- `outputs/catboost_gpu_final/models/catboost/best_model.joblib`
- `best_params.json`, `loss_history_per_epoch.json`, `trial_loss_history.json`
- `loss_curves_per_epoch.png`
- `predictions_*.png`, `scatter_*.png` for all 10 targets
- `test_metrics.json`

## 2. Neural Network Workflow (Completed)
**Script**: `run_nn_workflow_final.py`
**Supported Models**: MLP, RNN, GRU, LSTM, Transformer, Mamba

**Key Features**:
- MLP: Uses only current row data (no history)
- Sequence models (RNN/GRU/LSTM/Transformer/Mamba): Use history_length parameter
- Proper NaN handling in data loading
- Feature normalization with StandardScaler
- Single model, multi-output (unlike GBDT per-target models)

**Usage**:
```bash
# MLP (no history)
python run_nn_workflow_final.py --model MLP --n-trials 100 --output-dir outputs/nn_mlp

# RNN with history
python run_nn_workflow_final.py --model RNN --history-length 32 --n-trials 100 --output-dir outputs/nn_rnn
```

## 3. All NN Experiments Runner
**Script**: `run_all_nn_experiments.py`
Runs all 6 NN models sequentially with 100 trials each.

## 4. GBDT Experiment (XGBoost, LightGBM, CatBoost)
**Script**: `gbdt_experiment_final_v4.py`
Per-target models (10 separate models per algorithm)

**Usage**:
```bash
python gbdt_experiment_final_v4.py --n-trials 100 --output-dir outputs/gbdt_experiment
```

## Data Files
- **Input**: `data/imputed_data.csv` (11809 samples, 25 columns)
- **Input columns**: TRC-DT, TRC-RT, pH-DT, pH-RT, cond-DT, cond-RT, fDOM-RT, DO-RT, TOC-RT, DOC-RT
- **Output columns**: TRC-PPL1/2, pH-PPL1/2, cond-PPL1/2, TOC-PPL1/2, DOC-PPL1/2

## Environment
- **Conda env**: `torch`
- **Python**: 3.12
- **GPU**: NVIDIA GeForce RTX 4070 Laptop GPU
