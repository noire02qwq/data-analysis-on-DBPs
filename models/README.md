# Models Directory

This package contains PyTorch architectures (MLP, LSTM, RNN, GRU, Transformer, Mamba) and gradient-boosted learners (XGBoost, LightGBM, CatBoost), along with TOML configuration files used by the training scripts.

## Layout

| Path | Description |
| --- | --- |
| `mlp_regressor.py` | Multi-Layer Perceptron regressor |
| `lstm_regressor.py` | LSTM (Long Short-Term Memory) regressor |
| `rnn_regressor.py` | Simple RNN regressor |
| `gru_regressor.py` | GRU (Gated Recurrent Unit) regressor |
| `transformer_regressor.py` | Transformer encoder-only regressor |
| `mamba_regressor.py` | Mamba (State Space Model) regressor |
| `xgboost_regressor.py` | XGBoost wrapper for per-target training |
| `lightgbm_regressor.py` | LightGBM wrapper for per-target training |
| `catboost_regressor.py` | CatBoost wrapper for per-target training |
| `configs/` | TOML configuration files for all models |

## TOML Configuration Schema

Each config file contains three sections:

### `[model]` Section

Required keys:
- `type`: Model type (`MLP`, `LSTM`, `RNN`, `GRU`, `TRANSFORMER`, `MAMBA`, `XGBOOST`, `LIGHTGBM`, `CATBOOST`)
- `name`: Model name (used for output directory naming)

Model-specific parameters:
- **MLP**: `hidden_layers` (list), `dropout`
- **LSTM/RNN/GRU**: `units`, `num_layers`, `dropout`, `fc_dim` (optional)
- **Transformer**: `d_model`, `nhead`, `num_encoder_layers`, `dim_feedforward`, `dropout`, `fc_dim`
- **Mamba**: `d_model`, `n_layers`, `d_state`, `d_conv`, `expand`, `dropout`, `fc_dim`
- **XGBoost**: `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `gamma`, `reg_lambda`, `min_child_weight`
- **LightGBM**: `num_leaves`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_samples`, `reg_alpha`, `reg_lambda`, `bagging_freq`
- **CatBoost**: `depth`, `learning_rate`, `l2_leaf_reg`, `subsample`, `random_strength`, `bagging_temperature`

### `[training]` Section

- `max_epochs`: Maximum training epochs (or boosting rounds for GBDT)
- `batch_size`: Training batch size (for neural networks)
- `learning_rate`: Optimizer learning rate
- `weight_decay`: L2 regularization weight
- `patience`: Early stopping patience (0 to disable)
- `seed`: Random seed for reproducibility

### `[data]` Section

- `train_csv`: Path to training data CSV
- `val_csv`: Path to validation data CSV
- `test_csv`: Path to test data CSV
- `input_columns`: List of input feature column names
- `output_columns`: List of target column names

## Example Configuration Files

### RNN Config (`rnn_config.toml`)

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
weight_decay = 0.0
patience = 10
seed = 42

[data]
train_csv = "data/train.csv"
val_csv = "data/val.csv"
test_csv = "data/test.csv"
input_columns = ["TRC-DT", "pH-DT", "cond-DT"]
output_columns = ["TRC-PPL1", "TRC-PPL2"]
```

### Transformer Config (`transformer_config.toml`)

```toml
[model]
type = "TRANSFORMER"
name = "transformer_regressor"
history_length = 32
d_model = 128
nhead = 8
num_encoder_layers = 4
dim_feedforward = 512
dropout = 0.1

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

### Mamba Config (`mamba_config.toml`)

```toml
[model]
type = "MAMBA"
name = "mamba_regressor"
history_length = 32
d_model = 128
n_layers = 4
d_state = 16
d_conv = 4
expand = 2
dropout = 0.1

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

## Bayesian Optimization Configs

Bayesian optimization configs (`*_bayes.toml`) define the search space for hyperparameter tuning:

```toml
[parameters.history_length]
min = 32
max = 160
step = 8
log = false

[parameters.units]
min = 64
max = 384
step = 8
log = false

[parameters.dropout]
min = 0.1
max = 0.5
log = true

[parameters.batch_size]
min = 64
max = 384
step = 4
log = false

[parameters.learning_rate]
min = 0.0002
max = 0.002
log = true

[parameters.weight_decay]
min = 0.0001
max = 0.01
log = true
```

Use with `autotune.py`:

```bash
python scripts/autotune.py \
    --model-type RNN \
    --base-config models/configs/rnn_config.toml \
    --bayes-config models/configs/rnn_bayes.toml \
    --n-trials 100
```

## Comprehensive Experiment Script

The `run_comprehensive_experiment.py` script automates running all models with Bayesian optimization:

```bash
python scripts/run_comprehensive_experiment.py \
    --input data/imputed_data.csv \
    --output-dir outputs/comprehensive_experiment \
    --n-trials 100 \
    --models all
```

This script will:
1. Split data 70:15:15 (train:val:test)
2. Run Bayesian optimization (100 trials) for all 10 models
3. Train best model for each type
4. Generate test predictions and visualizations
5. Save all results to `outputs/comprehensive_experiment/`

Supported models:
- **GBDT**: XGBoost, LightGBM, CatBoost
- **Neural Networks**: MLP, RNN, GRU, LSTM, Mamba, Transformer
