# DBPs Regression Pipeline

End-to-end toolkit for preparing dissolved by-product sensor data, training multi-output regressors (MLP/LSTM/RNN/GRU/Transformer/Mamba/GBDT), with unified training, autotuning, and testing scripts.

## Overview

- **Data processing**: Impute missing values and align timestamps (using Polars)
- **Model zoo**: PyTorch models (MLP, LSTM, RNN, GRU, Transformer, Mamba) and tree ensembles (XGBoost, LightGBM, CatBoost)
- **Unified scripts**: Single `train.py`, `test.py`, `autotune.py` for all model types
- **Config format**: TOML configuration files
- **Comprehensive experiments**: Run all models with Bayesian optimization via `run_comprehensive_experiment.py`
- **Backend API**: Server on port 310 for web integration

## Repository Layout

| Path | Description |
| --- | --- |
| `data/` | Raw/imputed/aligned CSVs |
| `models/` | PyTorch modules and TOML configs |
| `scripts/` | CLI entry points for all tasks |
| `scripts/server.py` | Backend API server (port 310) |
| `scripts/demo_client.py` | Demo client (port 110) |
| `scripts/run_comprehensive_experiment.py` | Run all models with Bayes opt |

## Quick Start

### 1. Install Dependencies

```bash
pip install torch xgboost lightgbm catboost polars tomli tomli-w matplotlib optuna numpy
```

### 2. Prepare Data

```bash
# Split existing data into train/val/test
python scripts/split_data.py \
    --input data/time_aligned_data.csv \
    --train-rows 8000 \
    --val-rows 1500 \
    --test-rows 1500 \
    --output-dir data \
    --shuffle
```

### 3. Train a Model (e.g., RNN)

```bash
python scripts/train.py --config models/configs/rnn_config.toml
```

Output saved to `outputs/rnn_regressor/<timestamp>/`

### 4. Autotune Hyperparameters

```bash
python scripts/autotune.py \
    --model-type RNN \
    --base-config models/configs/rnn_config.toml \
    --bayes-config models/configs/rnn_bayes.toml \
    --n-trials 20
```

### 5. Test a Trained Model

```bash
python scripts/test.py --model-dir outputs/rnn_regressor/<timestamp>
```

### 6. Run Comprehensive Experiment (All Models)

```bash
python scripts/run_comprehensive_experiment.py \
    --input data/imputed_data.csv \
    --output-dir outputs/comprehensive_experiment \
    --n-trials 100 \
    --models all
```

This will:
1. Split data 70:15:15 (train:val:test)
2. Run Bayesian optimization (100 trials) for all models:
   - GBDT: XGBoost, LightGBM, CatBoost
   - Neural Networks: MLP, RNN, GRU, LSTM, Mamba, Transformer
3. Train best model for each type
4. Generate test predictions and visualizations
5. Save all results to `outputs/comprehensive_experiment/`

### 7. Process New Datasets (CAWW_35C, LSWW_29C, LSWW_35C)

```bash
python scripts/process_all_new_datasets.py
```

### 8. Fine-Tune Transformer on New Datasets

```bash
python scripts/fine_tune_transformer.py \
    --model-path outputs/transformer_regressor/<timestamp>/best_model.pt \
    --dataset caww_35c \
    --method full \
    --output-dir outputs/fine_tune_caww_35c
```

## Configuration Files

Model configs are in `models/configs/` with `.toml` extension:

- `*_config.toml` - Base model configuration
- `*_bayes.toml` - Bayesian optimization search space

### Example: RNN Config

```toml
[model]
type = "RNN"
name = "rnn_regressor"
history_length = 32
units = 128
num_layers = 2
dropout = 0.2

[training]
max_epochs = 100
batch_size = 128
learning_rate = 0.001
patience = 10

[data]
train_csv = "data/train.csv"
val_csv = "data/val.csv"
test_csv = "data/test.csv"
input_columns = ["TRC-DT", "pH-DT", "cond-DT"]
output_columns = ["TRC-PPL1", "TRC-PPL2"]
```

## Available Models

- **PyTorch**: MLP, LSTM, RNN, GRU, Transformer, Mamba
- **GBDT**: XGBoost, LightGBM, CatBoost

## Using the Backend Server

### Start Server (port 310)

```bash
python scripts/server.py --port 310
```

### API Endpoints

- `GET /health` - Health check
- `GET /models` - List available models
- `POST /split` - Split data
- `POST /train` - Train model
- `POST /autotune` - Run hyperparameter optimization
- `POST /test` - Test trained model

### Demo Client

```bash
python scripts/demo_client.py
```

## Data Format

Expected CSV columns include:
- Timestamp: `Date, Time`
- Sensors: `TRC-DT`, `TRC-RT`, `TRC-PPL1`, `TRC-PPL2`, `pH-DT`, `pH-RT`, `pH-PPL1`, `pH-PPL2`, etc.

## Project Structure

```
.
├── data/                    # Data files
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── models/                  # Model implementations
│   ├── configs/             # TOML configs
│   └── *.py                 # Model classes
├── scripts/                 # CLI scripts
│   ├── train.py            # Training script
│   ├── test.py             # Testing script
│   ├── autotune.py         # Hyperparameter optimization
│   ├── split_data.py       # Data splitting
│   ├── server.py           # API server
│   ├── demo_client.py      # Demo client
│   └── run_comprehensive_experiment.py  # Run all models
├── outputs/                 # Training outputs
└── README.md
```

## Documentation

See `scripts/README.md` for detailed CLI documentation.
