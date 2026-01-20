# DBPS Regression Pipeline

End-to-end toolkit for preparing dissolved by-product sensor data, training multi-output regressors (MLP/LSTM/RNN/GBDT), and comparing absolute-value vs. rate-based targets.

## Overview
- **Data processing**: impute missing `-PPL1/-PPL2` readings and realign DT/RT/PPL1/PPL2 timestamps.
- **Model zoo**: PyTorch dense/sequence models and tree ensembles (XGBoost, LightGBM, CatBoost) with shared training/test scripts.
- **Rate mode**: optional `(PPL-RT)/RT` objective during training, with automatic conversion back to absolute values for metrics.
- **Autotuning**: Bayesian optimization runner that explores hyperparameter spaces efficiently (replacing the old grid/hill-climbing approach).
- **Archival**: `Archived/` stores the top checkpoints + metadata from major sweeps for future reference.

## Repository Layout
| Path | Description |
| --- | --- |
| `data/` | Raw/imputed/aligned CSVs. Training scripts default to `data/time_aligned_data.csv`. |
| `models/` | PyTorch modules, gradient-boost wrappers, and all YAML configs & grids (see `models/README.md`). |
| `scripts/` | CLI entry points for data prep, training, evaluation, and autotuning (documented in `scripts/README.md`). |
| `Archived/` | Snapshots of best autotune runs, including `metadata.json`, scalers, and checkpoints. |
| `prompt-draft/` | Narrative specs for past experiments (kept for context). |
| `requirements.txt` | Python dependencies (NumPy/Pandas/Torch/XGBoost/LightGBM/CatBoost, etc.). |

## Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
> CatBoost and LightGBM wheels are included, so no extra system packages are required on Linux.

## Data Preparation Workflow
1. **Impute missing values**:
   ```bash
   python scripts/fill_missing.py \
       --input data/raw_data.csv --output data/imputed_data.csv
   ```
2. **Align timestamps** (shifts RT/PPL columns to a common DT reference):
   ```bash
   python scripts/align_time.py \
       --input data/imputed_data.csv --output data/time_aligned_data.csv
   ```
Both scripts log the decisions they make; rerun them whenever you update the source CSVs.

## Training & Testing Overview
- **Value-domain models** use `scripts/train_regression.py` / `scripts/test_regression.py`.
- **Rate-domain models** use `scripts/rate_train_regression.py` / `scripts/rate_test_regression.py`; they consume the same YAML configs but change the target transformation.
- Supported types: `MLP`, `MLP_WITH_HISTORY`, `LSTM`, `RNN`, `XGBOOST`, `LIGHTGBM`, `CATBOOST`.
- All scripts expect a YAML config under `models/configs/` specifying the model, training hyperparameters, and data paths.

### Example: LSTM Workflow
1. **Inspect config** (`models/configs/lstm_config.yaml`):
   ```yaml
   model:
     type: LSTM
     name: lstm_regressor
     history_length: 48
     units: 192
     num_layers: 4
     dropout: 0.25
     fc_dim: 128
   training:
     max_epochs: 600
     batch_size: 256
     learning_rate: 0.001
     weight_decay: 0.0
     seed: 2023
   data:
     input_csv: data/time_aligned_data.csv
     timestamp_column: "Date, Time"
   ```
2. **Train (value domain)**:
   ```bash
   python scripts/train_regression.py --config models/configs/lstm_config.yaml
   ```
   Artifacts appear in `scripts/outputs/lstm_regressor/` (metadata, scalers, checkpoints, plots).
3. **Test**:
   ```bash
   python scripts/test_regression.py --model-dir scripts/outputs/lstm_regressor
   ```
   The script reloads the saved scalers, rebuilds the splits, and writes `test_predictions.csv` plus per-target plots.
4. **Autotune** (Bayesian Optimization):
   ```bash
   python scripts/bayes_autotune_trc.py \
       --model-type LSTM \
       --base-config models/configs/lstm_config.yaml \
       --grid-config models/configs/lstm_bayes.yaml \
       --n-trials 50
   ```
   Each run lands in `scripts/outputs/lstm-trc-value/`; `optuna_study.db` (or similar logs) and `autotune_results.csv` record the progress.
5. **Rate variant**: use `scripts/bayes_autotune_trc_rate.py` (or `_other_rate.py` for non-TRC) and point to the same configs.

## Autotuning Summary
We now use **Bayesian Optimization** (via Optuna or similar logic) to find optimal hyperparameters, which is more efficient than grid search.

| Target | Value Domain Script | Rate Domain Script |
| --- | --- | --- |
| **TRC** (Chlorine) | `scripts/bayes_autotune_trc.py` | `scripts/bayes_autotune_trc_rate.py` |
| **Other** (pH/TOC/etc) | `scripts/bayes_autotune_other.py` | `scripts/bayes_autotune_other_rate.py` |

**Usage**:
All scripts share the same flags:
- `--model-type`: `LSTM`, `RNN`, `MLP`, `XGBOOST`, `LIGHTGBM`, `CATBOOST`, etc.
- `--base-config`: path to the default YAML (e.g., `models/configs/xgboost_config.yaml`).
- `--grid-config`: path to the search space YAML (e.g., `models/configs/xgboost_bayes.yaml`).

**Bayesian logic**: The optimizer builds a probabilistic model of the objective function (validation MSE) and selects the next hyperparameters to balance exploration (uncertain regions) and exploitation (promising regions). output is logged to `autotune_results.csv`.

## Archived Best Runs
Whenever an autotune sweep finishes, copy the best-performing folder into `Archived/<run_id><model>/` (see `Archived/README.md`). These archives preserve the exact environment required to reproduce research results:
- Re-run tests: `python scripts/test_regression.py --model-dir Archived/0048lstm`
- Compare configs/metrics across experiments.

## Documentation & References
- `scripts/README.md`: detailed CLI reference for every step.
- `models/README.md`: explanation of model modules and YAML schema.
- `requirements.txt`: dependency pins.

For experimental context (business logic, historical notes), consult the relevant markdown files under `prompt-draft/`. The codebase itself remains agnostic to individual experiment IDs—everything is configured through the YAML files and CLI commands outlined above.
