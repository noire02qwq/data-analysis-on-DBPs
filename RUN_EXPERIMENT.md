# How to Run the Comprehensive Experiment

## Prerequisites

1. Activate the conda environment:
```bash
conda activate torch
```

2. Verify dependencies are installed:
```bash
python -c "import torch, xgboost, lightgbm, catboost, optuna, polars; print('All dependencies installed')"
```

## Running the Experiment

### Option 1: Run All Models (Recommended)

```bash
python scripts/run_comprehensive_experiment.py \
    --input data/imputed_data.csv \
    --output-dir outputs/comprehensive_experiment \
    --n-trials 100 \
    --models xgboost lightgbm catboost mlp rnn gru lstm transformer mamba
```

### Option 2: Run Individual Models

```bash
# XGBoost only
python scripts/run_comprehensive_experiment.py \
    --input data/imputed_data.csv \
    --output-dir outputs/comprehensive_experiment \
    --n-trials 100 \
    --models xgboost

# Neural networks only
python scripts/run_comprehensive_experiment.py \
    --input data/imputed_data.csv \
    --output-dir outputs/comprehensive_experiment \
    --n-trials 100 \
    --models mlp rnn gru lstm transformer mamba
```

### Option 3: Quick Test (2 trials per model)

```bash
python scripts/run_comprehensive_experiment.py \
    --input data/imputed_data.csv \
    --output-dir outputs/test_run \
    --n-trials 2 \
    --models xgboost
```

## Monitoring Progress

1. Check the output directory:
```bash
ls -la outputs/comprehensive_experiment/
```

2. Check individual model progress:
```bash
ls outputs/comprehensive_experiment/models/
```

3. Check experiment summary:
```bash
cat outputs/comprehensive_experiment/experiment_summary.json
```

## Resume After Interruption

The script automatically skips completed models. Just re-run the same command:

```bash
python scripts/run_comprehensive_experiment.py \
    --input data/imputed_data.csv \
    --output-dir outputs/comprehensive_experiment \
    --n-trials 100 \
    --models xgboost lightgbm catboost mlp rnn gru lstm transformer mamba
```

## Expected Runtime

- XGBoost/LightGBM/CatBoost: ~30-60 minutes per model (100 trials)
- MLP/RNN/GRU/LSTM: ~2-4 hours per model (100 trials)
- Transformer/Mamba: ~3-6 hours per model (100 trials)

Total estimated time: ~20-40 hours for all 10 models with 100 trials each.

## Output Structure

```
outputs/comprehensive_experiment/
├── data_split/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── models/
│   ├── xgboost/
│   │   ├── bayes_opt/           # Bayesian optimization trials
│   │   ├── best_config.toml     # Best hyperparameters
│   │   └── final_model/         # Trained model
│   ├── lightgbm/
│   ├── ... (other models)
│   └── mamba/
└── experiment_summary.json      # Overall results
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch_size in the bayes config files
- Use smaller model configurations
- Train fewer models simultaneously

### Slow Training
- Ensure GPU is being used: `python -c "import torch; print(torch.cuda.is_available())"`
- Reduce n-trials for quicker results
- Use fewer models

### Import Errors
```bash
pip install optuna tomli tomli-w pyyaml polars matplotlib
```

## Support

For issues or questions, please refer to:
- `CLAUDE.md` - Project documentation
- `IMPLEMENTATION_SUMMARY.md` - Implementation details
- `models/README.md` - Model documentation
- `scripts/README.md` - Script usage guide