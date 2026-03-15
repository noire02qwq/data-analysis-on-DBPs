# Scripts Usage Guide

This folder contains all CLI entry points for the DBPs data-processing pipeline.

## Available Scripts

| Script | Description |
|--------|-------------|
| `split_data.py` | Split CSV data into train/val/test sets |
| `train.py` | Train regression models |
| `test.py` | Test trained models |
| `autotune.py` | Bayesian hyperparameter optimization |
| `server.py` | Backend API server (port 310) |
| `demo_client.py` | Demo client for testing workflow |

## 1. Data Splitting (`split_data.py`)

Split a CSV file into train, validation, and test sets.

```bash
python scripts/split_data.py \
    --input data/time_aligned_data.csv \
    --train-rows 8000 \
    --val-rows 1500 \
    --test-rows 1500 \
    --output-dir data \
    --shuffle \
    --seed 42
```

**Options:**
- `--input`: Input CSV file (required)
- `--train-rows`: Number of training rows (required)
- `--val-rows`: Number of validation rows (required)
- `--test-rows`: Number of test rows (required)
- `--output-dir`: Output directory (required)
- `--shuffle`: Randomly shuffle before splitting
- `--seed`: Random seed (default: 42)

## 2. Model Training (`train.py`)

Train a regression model using TOML configuration.

```bash
python scripts/train.py --config models/configs/rnn_config.toml
```

**Options:**
- `--config`: Path to TOML config file (required)

**Supported Models:**
- PyTorch: MLP, LSTM, RNN, GRU, Transformer
- GBDT: XGBoost, LightGBM, CatBoost

**Config Format:**
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
seed = 42

[data]
train_csv = "data/train.csv"
val_csv = "data/val.csv"
test_csv = "data/test.csv"
input_columns = ["TRC-DT", "pH-DT", "cond-DT"]
output_columns = ["TRC-PPL1", "TRC-PPL2"]
```

**Output:**
- `outputs/<model_name>/<timestamp>/` directory containing:
  - `best_model.pt` / `best_model_*.xgb` - Best model checkpoint
  - `last_model.pt` - Final model checkpoint
  - `scalers.npz` - Feature scalers
  - `config.toml` - Copy of configuration
  - `result.toml` - Training results
  - `loss_history.csv` - Training history
  - `training_curve.png` - Loss visualization

## 3. Model Testing (`test.py`)

Test a trained model on test data.

```bash
python scripts/test.py --model-dir outputs/rnn_regressor/<timestamp>
```

**Options:**
- `--model-dir`: Directory containing trained model (required)
- `--test-csv`: Optional test CSV override
- `--output-dir`: Optional output directory

**Output:**
- `test_metrics.csv` - MSE, RMSE, MAE, R² metrics
- `test_predictions.csv` - Predicted values
- `*_pred_vs_true.png` - Prediction vs true plots

## 4. Autotuning (`autotune.py`)

Run Bayesian optimization for hyperparameter tuning.

```bash
python scripts/autotune.py \
    --model-type RNN \
    --base-config models/configs/rnn_config.toml \
    --bayes-config models/configs/rnn_bayes.toml \
    --n-trials 20
```

**Options:**
- `--model-type`: Model type (required) - MLP, LSTM, RNN, GRU, TRANSFORMER, XGBOOST, LIGHTGBM, CATBOOST
- `--base-config`: Base configuration TOML file (required)
- `--bayes-config`: Bayesian search space TOML file (required)
- `--n-trials`: Number of optimization trials (default: 20)
- `--output-dir`: Output directory (optional)
- `--study-name`: Optuna study name (optional)
- `--storage`: Optuna storage URL (optional, e.g., sqlite:///study.db)

**Bayes Config Format:**
```toml
[model]
history_length = {min = 16, max = 64}
units = {min = 64, max = 256}
num_layers = {min = 1, max = 4}
dropout = {min = 0.1, max = 0.4}

[training]
batch_size = {min = 64, max = 256}
learning_rate = {min = 0.0001, max = 0.01, log = true}
```

## 5. API Server (`server.py`)

Run the backend API server for web integration.

```bash
python scripts/server.py --port 310
```

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/models` | GET | List available models |
| `/split` | POST | Split data |
| `/train` | POST | Train model |
| `/autotune` | POST | Run autotuning |
| `/test` | POST | Test model |

**Request Format:**
```json
{
    "input_csv": "data/time_aligned_data.csv",
    "train_rows": 8000,
    "val_rows": 1500,
    "test_rows": 1500,
    "output_dir": "data",
    "shuffle": true,
    "seed": 42
}
```

## 6. Demo Client (`demo_client.py`)

Run a demo that tests the full workflow.

```bash
python scripts/demo_client.py
```

This runs: data check → split (if needed) → train → autotune → test

## Typical Workflow

```bash
# 1. Split data (if not already done)
python scripts/split_data.py \
    --input data/time_aligned_data.csv \
    --train-rows 8000 \
    --val-rows 1500 \
    --test-rows 1500 \
    --output-dir data \
    --shuffle

# 2. Train a model
python scripts/train.py --config models/configs/rnn_config.toml

# 3. Autotune (optional)
python scripts/autotune.py \
    --model-type RNN \
    --base-config models/configs/rnn_config.toml \
    --bayes-config models/configs/rnn_bayes.toml \
    --n-trials 20

# 4. Test the model
python scripts/test.py --model-dir outputs/rnn_regressor/<timestamp>
```