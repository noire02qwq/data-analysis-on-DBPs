# DBPs Regression Pipeline

End-to-end toolkit for preparing dissolved by-product sensor data, training multi-output regressors (MLP/LSTM/RNN/GRU/Transformer/Mamba/GBDT), with unified training, autotuning, and testing scripts.

## Overview

- **Environment**: uv (fast Python package manager) with PyTorch 2.5 + CUDA 12.1
- **Data processing**: Impute missing values and align timestamps (using Polars)
- **Model zoo**: PyTorch models (MLP, LSTM, RNN, GRU, Transformer, Mamba) and tree ensembles (XGBoost, LightGBM, CatBoost)
- **Unified scripts**: Single `train.py`, `test.py`, `autotune.py` for all model types
- **Config format**: TOML configuration files
- **Backend API**: Flask server on port 5555 for web integration

## Repository Layout

| Path | Description |
| --- | --- |
| `data/` | Raw/imputed/aligned CSVs |
| `models/` | PyTorch modules and TOML configs |
| `scripts/` | CLI entry points for all tasks |
| `backend_server.py` | Backend API server (port 5555) |
| `outputs/` | Training outputs and results |

## Quick Start

### 1. Install Dependencies (uv)

```bash
# Install uv if not available
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment with Python 3.12
uv venv .venv --python 3.12
source .venv/bin/activate

# Install PyTorch with CUDA
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install all dependencies
uv pip install flask flask-cors polars scikit-learn xgboost lightgbm catboost optuna tomli tomli-w matplotlib python-dateutil openpyxl
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

## Using the Backend API Server

### Start Server (port 5555)

```bash
python backend_server.py --port 5555
```

### API Endpoints

- `GET /health` - Health check
- `POST /api/v1/data/upload` - Upload dataset
- `POST /api/v1/data/split` - Split dataset
- `POST /api/v1/train` - Train model
- `GET /api/v1/train/<id>/status` - Get training status
- `POST /api/v1/train/<id>/stop` - Stop training
- `POST /api/v1/tune` - Run hyperparameter optimization
- `POST /api/v1/test` - Test trained model
- `POST /api/v1/predict` - Run prediction
- `GET /api/v1/models` - List trained models
- `GET /api/v1/models/<id>` - Get model details
- `DELETE /api/v1/models/<id>` - Delete model
- `GET /api/v1/models/<id>/download` - Download model

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
│   └── run_comprehensive_experiment.py  # Run all models
├── outputs/                 # Training outputs
├── backend_server.py        # Flask API server
├── pyproject.toml           # uv project configuration
├── Dockerfile               # Docker container (GPU)
├── Dockerfile.cpu           # Docker container (CPU only)
└── README.md
```

## Documentation

See `scripts/README.md` for detailed CLI documentation.

## Docker Deployment

### Build Image

```bash
# GPU version (with CUDA support)
docker build -t dbps-backend:latest .

# CPU-only version
docker build -f Dockerfile.cpu -t dbps-backend:cpu .
```

### Run Container

```bash
# GPU version
docker run -d --gpus all \
  -p 5555:5555 \
  -v $(pwd)/data:/app/data:ro \
  -v dbps-outputs:/app/outputs \
  --name dbps-backend \
  dbps-backend:latest

# CPU version
docker run -d \
  -p 5555:5555 \
  -v $(pwd)/data:/app/data:ro \
  -v dbps-outputs:/app/outputs \
  --name dbps-backend \
  dbps-backend:cpu
```

### Using Docker Compose (Recommended)

From the project root:
```bash
docker-compose -f docker-compose.yml up backend

# Or CPU-only
docker-compose -f docker-compose.yml -f docker-compose.cpu.yml up backend
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHONUNBUFFERED` | 1 | Unbuffered Python output |
| `FLASK_ENV` | production | Flask environment |
| `CUDA_VISIBLE_DEVICES` | "" (empty) | GPU device IDs (set for CPU-only) |

### Ports

| Service | Port | Description |
|---------|------|-------------|
| Backend API | 5555 | Flask REST API |

### Volumes

| Volume | Mount Point | Description |
|--------|------------|-------------|
| dbps-outputs | /app/outputs | Trained model outputs |
| dbps-uploads | /app/uploads | Temporary upload files |
| ./data | /app/data:ro | Dataset directory (read-only) |
