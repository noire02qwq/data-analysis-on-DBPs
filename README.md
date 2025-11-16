# data-analysis-on-DBPs

End-to-end workflow for preparing dissolved by-product sensor data, aligning time steps, and running the 回归实验1109 double-output regressions (TRC/TOC/DOC/pH at PPL1/PPL2) using PyTorch (MLP/MLP-with-history/LSTM/RNN) or XGBoost.

## Repository Layout

| Path | Description |
| --- | --- |
| `data/` | Raw and processed CSV files (`raw_data.csv`, `imputed_data.csv`, `time_aligned_data.csv`). |
| `models/` | PyTorch modules for the MLP/MLP-with-history/LSTM/RNN architectures plus YAML configs under `models/configs/`. |
| `prompt-draft/` | Vibe-coding task briefs that describe each experiment in natural language. |
| `scripts/` | All runnable scripts (`fill_missing.py`, `align_time.py`, `train_regression.py`, `test_regression.py`, shared regression utilities, plus a script-level README) and generated artifacts in `scripts/outputs/<model_name>/`. |
| `requirements.txt` | Python dependencies for running every step. |

## Installing Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Typical Workflow

1. **Fill missing values**  
   `python scripts/fill_missing.py`
2. **Align time steps**  
   `python scripts/align_time.py`
3. **Train a model** (pick any YAML config under `models/configs/`)  
   `python scripts/train_regression.py --config models/configs/mlp_config.yaml`  
   `python scripts/train_regression.py --config models/configs/mlp_with_history_config.yaml`  
   `python scripts/train_regression.py --config models/configs/lstm_config.yaml`  
   `python scripts/train_regression.py --config models/configs/rnn_config.yaml`  
   `python scripts/train_regression.py --config models/configs/xgboost_config.yaml`
4. **Evaluate / plot predictions**  
   `python scripts/test_regression.py --model-dir scripts/outputs/<model_name>`

Detailed script options and outputs are documented in `scripts/README.md`.

## Rate-Based Regression (变化率回归1115)

The `变化率回归1115.md` brief swaps each target from an absolute value to the RT-relative delta `((PPL - RT) / RT)` so that models learn smoother distributions but final metrics remain in the original value domain. This repository now ships a parallel set of scripts prefixed with `rate_`:

| Step | Command |
| --- | --- |
| Train | `python scripts/rate_train_regression.py --config models/configs/<config>.yaml` |
| Test | `python scripts/rate_test_regression.py --model-dir scripts/outputs/rate_<model_name>` |
| Autotune LSTM | `nohup python scripts/rate_autotune_lstm.py > rate_autotune_lstm.log 2>&1 &` |
| Autotune RNN | `nohup python scripts/rate_autotune_rnn.py  > rate_autotune_rnn.log 2>&1 &` |

The rate-aware scripts keep the same YAML configs and model code as the direct-regression pipeline, automatically prefix their output folders with `rate_`, and always convert predictions back to absolute values before computing validation/test MSE.

## Autotuning (optional)

Once any baseline run is healthy, you can launch the grid searches described in `prompt-draft/回归实验1109.md`:

```bash
# LSTM grid search
nohup python scripts/autotune_lstm.py > autotune_lstm.log 2>&1 &

# RNN grid search
nohup python scripts/autotune_rnn.py > autotune_rnn.log 2>&1 &

# MLP / MLP-with-history (value domain)
nohup python scripts/autotune_mlp.py > autotune_mlp.log 2>&1 &
nohup python scripts/autotune_mlp_history.py > autotune_mlp_history.log 2>&1 &

# Rate-based counterparts
nohup python scripts/rate_autotune_lstm.py > rate_autotune_lstm.log 2>&1 &
nohup python scripts/rate_autotune_rnn.py > rate_autotune_rnn.log 2>&1 &
nohup python scripts/rate_autotune_mlp.py > rate_autotune_mlp.log 2>&1 &
nohup python scripts/rate_autotune_mlp_history.py > rate_autotune_mlp_history.log 2>&1 &
```

Each script enumerates the ranges from its paired grid file (`models/configs/lstm_grid.yaml`, `rnn_grid.yaml`, `mlp_grid.yaml`, or `mlp_with_history_grid.yaml`), writes numbered folders (e.g., `scripts/outputs/lstm_regressor/0001/`), and appends every run’s hyperparameters + best validation loss to `scripts/outputs/<model>/autotune_results.csv`. Use `--max-trials` or `--start-index` if you want to split the workload across multiple nights.

## Notes

- Install the requirements inside a virtual environment (`pip install -r requirements.txt`) to get PyTorch + the pinned XGBoost build (2.0.x keeps compatibility with glibc < 2.28).  
- `prompt-draft/` files describe the business logic (缺失填充、时间对齐、回归实验10xx) that each script implements.  
- The training script no longer writes periodic checkpoints; it logs each epoch’s train/val MSE, only starts tracking “best” checkpoints after the train loss falls below one quarter of its initial value, and enforces the updated stopping rule (50-epoch patience beginning after epoch 30, force stop at epoch 120 if needed). XGBoost trains eight independent single-output boosters, each with its own MSE loss history/plot under `scripts/outputs/<model_name>/`.  
- Every training run copies its config plus metadata, scalers, checkpoints, and plots into `scripts/outputs/<model_name>/`, allowing `scripts/test_regression.py` to rebuild the splits and evaluate consistently.  
- The autotuning scripts mentioned in 回归实验1109 are intentionally postponed; once the five core models are stable they can be added on top of this structure.
