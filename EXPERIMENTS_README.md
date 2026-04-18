# DBPs Regression Experiments

This document describes how to run the complete set of experiments for the DBPs regression pipeline.

## Overview

The experiments include:
- **GBDT Models**: XGBoost, LightGBM, CatBoost (per-target models)
- **Neural Networks**: MLP, RNN, GRU, LSTM, Transformer, Mamba (single multi-output models)

## Prerequisites

```bash
conda activate torch
```

Ensure you have:
- PyTorch with CUDA support
- Optuna for Bayesian optimization
- XGBoost, LightGBM, CatBoost
- Polars, NumPy, Matplotlib
- scikit-learn (for StandardScaler and metrics)

## Project Structure

```
├── run_catboost_gpu_only.py       # CatBoost GPU experiment (standalone)
├── run_nn_workflow_final.py       # Neural Network workflow (all 6 models)
├── run_all_nn_experiments.py      # Runner for all NN experiments
├── gbdt_experiment_final_v4.py  # GBDT experiments (XGB/LGB/Cat)
├── models/                         # Neural network model definitions
│   ├── mlp_regressor.py
│   ├── lstm_regressor.py
│   ├── rnn_regressor.py
│   ├── gru_regressor.py
│   ├── transformer_regressor.py
│   └── mamba_regressor.py
└── outputs/                        # Experiment results
```

## Data Preparation

Data should be in `data/imputed_data.csv` with the following columns:
- **Inputs**: TRC-DT, TRC-RT, pH-DT, pH-RT, cond-DT, cond-RT, fDOM-RT, DO-RT, TOC-RT, DOC-RT
- **Outputs**: TRC-PPL1, TRC-PPL2, pH-PPL1, pH-PPL2, cond-PPL1, cond-PPL2, TOC-PPL1, TOC-PPL2, DOC-PPL1, DOC-PPL2

## Running Experiments

### 1. GBDT Models

#### XGBoost
```bash
python gbdt_experiment_final_v4.py --model xgboost --n-trials 100 --output-dir outputs/gbdt_xgboost
```

#### LightGBM
```bash
python gbdt_experiment_final_v4.py --model lightgbm --n-trials 100 --output-dir outputs/gbdt_lightgbm
```

#### CatBoost (GPU) - Fixed Version
```bash
python run_catboost_gpu_only.py --n-trials 100 --output-dir outputs/catboost_gpu
```

**Note on CatBoost Fix**: The original script had an error:
```
Error: default bootstrap type (bayesian) doesn't support 'subsample' option
```

This was fixed by using `bootstrap_type: 'Poisson'` in the CatBoost parameters. The Poisson bootstrap supports `subsample < 1.0` and works correctly with GPU training.

**Key differences from GBDT script**:
- This is a standalone script (only runs CatBoost)
- Uses Poisson bootstrap which is faster on GPU
- Same output structure as GBDT v4 script

### 2. Neural Network Models

All NN models use a unified workflow in `run_nn_workflow_final.py`:

#### MLP (Multi-Layer Perceptron)
- Uses only current row data (no history/sequences)
- Single model, multi-output
```bash
python run_nn_workflow_final.py --model MLP --n-trials 100 --output-dir outputs/nn_mlp
```

#### RNN / GRU / LSTM
- Use sequences with `history_length` parameter
- Single model, multi-output
```bash
python run_nn_workflow_final.py --model RNN --n-trials 100 --history-length 32 --output-dir outputs/nn_rnn
python run_nn_workflow_final.py --model GRU --n-trials 100 --history-length 32 --output-dir outputs/nn_gru
python run_nn_workflow_final.py --model LSTM --n-trials 100 --history-length 32 --output-dir outputs/nn_lstm
```

#### Transformer / Mamba
- Use sequences with `history_length` parameter
- Single model, multi-output
```bash
python run_nn_workflow_final.py --model TRANSFORMER --n-trials 100 --history-length 32 --output-dir outputs/nn_transformer
python run_nn_workflow_final.py --model MAMBA --n-trials 100 --history-length 32 --output-dir outputs/nn_mamba
```

#### Run All NN Experiments
```bash
python run_all_nn_experiments.py
```
This runs all 6 NN models (MLP, RNN, GRU, LSTM, Transformer, Mamba) sequentially with 100 trials each.

## Output Structure

All experiments produce the following output structure:

```
outputs/<experiment_name>/
├── models/
│   └── <model_name>/
│       ├── best_model.pt (or .joblib for GBDT)
│       ├── best_params.json
│       ├── loss_history_per_epoch.json
│       ├── loss_curves_per_epoch.png
│       ├── test_metrics.json
│       ├── trial_loss_history.json
│       ├── predictions_<target>.png (per target)
│       └── scatter_<target>.png (per target)
└── data_split/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

## Key Differences Between Model Types

### GBDT Models (XGBoost, LightGBM, CatBoost)
- **Training**: One model per target (10 separate models)
- **Input**: Current row features only
- **GPU**: CatBoost supports GPU; XGBoost and LightGBM use CPU
- **Output**: Individual models per target

### MLP (Multi-Layer Perceptron)
- **Training**: Single model, multi-output (10 outputs)
- **Input**: Current row features only (flattened)
- **GPU**: Yes
- **Output**: Single model with 10 outputs

### Sequence Models (RNN, GRU, LSTM, Transformer, Mamba)
- **Training**: Single model, multi-output (10 outputs)
- **Input**: Sequences with `history_length` (sliding window)
- **GPU**: Yes
- **Output**: Single model with 10 outputs

## Troubleshooting

### CatBoost GPU Issues
If you encounter:
```
Error: default bootstrap type (bayesian) doesn't support 'subsample' option
```

Make sure to use the fixed script: `run_catboost_gpu_fixed.py`

### Neural Network NaN Values
If training produces NaN values:
- Gradient clipping is implemented in `run_nn_workflow.py` to prevent this
- If NaN persists, try reducing learning rate or increasing batch size

### Out of Memory (OOM)
If you encounter OOM errors:
- Reduce `batch_size` in the Bayesian optimization
- Reduce `history_length` for sequence models
- Use smaller model architectures (fewer layers, smaller hidden dimensions)

## Notes

- All scripts automatically use GPU if available
- Bayesian optimization uses 100 trials by default (adjust with `--n-trials`)
- Random seed is set to 42 by default for reproducibility
- All outputs are saved in the specified `--output-dir`
