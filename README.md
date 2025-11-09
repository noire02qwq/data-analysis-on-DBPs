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

## Notes

- Install the requirements inside a virtual environment (`pip install -r requirements.txt`) to get PyTorch + the pinned XGBoost build (2.0.x keeps compatibility with glibc < 2.28).  
- `prompt-draft/` files describe the business logic (缺失填充、时间对齐、回归实验10xx) that each script implements.  
- The training script checkpoints every 10 epochs, logs each epoch’s train MSE / val MAE, and only starts tracking “best” checkpoints after the train loss falls below one quarter of its initial value; early stopping also waits for that same threshold plus at least 80 epochs. XGBoost now trains eight independent single-output boosters, each with its own loss history/plot under `scripts/outputs/<model_name>/`.  
- Every training run copies its config plus metadata, scalers, checkpoints, and plots into `scripts/outputs/<model_name>/`, allowing `scripts/test_regression.py` to rebuild the splits and evaluate consistently.  
- The autotuning scripts mentioned in 回归实验1109 are intentionally postponed; once the five core models are stable they can be added on top of this structure.
