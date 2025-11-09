# Scripts Usage Guide

This folder contains every step of the DBPS data-processing pipeline, from filling missing values all the way to training and evaluating regression models. The sections below explain what each script does, the required inputs, and the most relevant flags.

## 1. Missing-Value Imputation (`fill_missing.py`)
- **Purpose:** Fills all `-PPL1/-PPL2` columns in `data/raw_data.csv` according to the interpolation rules described in `prompt-draft/缺失填充.md`.
- **Default command:**  
  ```bash
  python scripts/fill_missing.py
  ```
- **Key options:**  
  - `--input`: path to the original CSV (default `data/raw_data.csv`).  
  - `--output`: destination for the filled table (default `data/imputed_data.csv`).  
  - `--seed`: random seed used when sampling the variance-based noise term (default `42`).
- **Outputs:** Saves the filled CSV to `data/imputed_data.csv` (or the path you specify) and prints the number of synthesized values for each column.

## 2. Time Alignment (`align_time.py`)
- **Purpose:** Removes the empty `cond-RT` column and realigns the DT/RT/PPL1/PPL2 measurements so that each row represents the same fluid batch. The offsets follow `prompt-draft/时间对齐.md`: RT is shifted +25h, PPL1 +38h, and PPL2 +55h relative to DT.
- **Default command:**  
  ```bash
  python scripts/align_time.py
  ```
- **Key options:**  
  - `--input`: imputed CSV to align (default `data/imputed_data.csv`).  
  - `--output`: aligned CSV (default `data/time_aligned_data.csv`).  
  - `--timestamp-column`: column containing the DT timestamp (default `"Date, Time"`).
- **Outputs:** Aligned CSV in `data/time_aligned_data.csv`, logs dropped rows, and removes `cond-RT`.

## 3. Regression Training (`train_regression.py`)
- **Purpose:** Implements the 回归实验1109 specification: trains MLP / MLP_WITH_HISTORY / LSTM / RNN PyTorch models or an XGBoost regressor to predict `TRC/TOC/DOC/pH` at PPL1/PPL2 using only non-PPL features.
- **Command template:**  
  ```bash
  python scripts/train_regression.py --config models/configs/<config>.yaml
  ```
- **Bundled configs (feel free to copy & edit):**  
  - `mlp_config.yaml`: single-step MLP without history.  
  - `mlp_with_history_config.yaml`: flattened history window MLP.  
  - `lstm_config.yaml`: stacked LSTM fed with non-PPL sequences.  
  - `rnn_config.yaml`: SimpleRNN counterpart.  
  - `xgboost_config.yaml`: gradient-boosted regressor.  
- **Training behaviour:**  
  - Splits data 70/15/15, scales features with the train slice, and removes all `-PPL1/-PPL2` inputs.  
  - Saves a checkpoint every 10 epochs/rounds (but logs every epoch’s train MSE + val MAE). “Best” checkpoints start updating only after the train loss falls below 其初始值的1/4，早停计数也是从这一刻（且至少80轮）才开始统计；若100轮后仍未达到1/4阈值将直接提前结束。  
  - PyTorch models output `best_model.pt` / `last_model.pt` plus a single `training_curve.png` + `loss_history.csv`. XGBoost现在会顺序训练8个单输出的 booster，每个目标各自产生 `training_curve_<target>.png` / `loss_history_<target>.csv` 以及对应的 best/last `*.json`。  
  - 每次训练还会写入 `metadata.json`, `scalers.npz` 以及原始 YAML 副本到 `scripts/outputs/<model_name>/`。

## 4. Regression Testing (`test_regression.py`)
- **Purpose:** Loads the artifacts referenced in `metadata.json`, rebuilds the exact dataset splits, evaluates the best model on the test slice, and emits per-target plots plus a CSV (works for both PyTorch checkpoints and XGBoost boosters).
- **Command template:**  
  ```bash
  python scripts/test_regression.py --model-dir scripts/outputs/mlp_regressor
  ```
- **Key options:**  
  - `--model-dir`: folder holding `metadata.json`, `scalers.npz`, and the saved best model.  
  - `--data`: optional override CSV; otherwise it uses the path recorded in `metadata.json`.
- **Outputs:**  
  - Per-target plots (e.g., `TRC-PPL1_pred_vs_true.png`).  
  - `test_predictions.csv` with paired true/predicted values.  
  - Console summary of per-target MSE.

## Typical Workflow
1. `python scripts/fill_missing.py`  
2. `python scripts/align_time.py`  
3. `python scripts/train_regression.py --config models/configs/<your_config>.yaml`  
4. `python scripts/test_regression.py --model-dir scripts/outputs/<model_name>`

This sequence ensures the regression models always see the cleaned, aligned data and that evaluation uses the same scalers and splits recorded during training. Adjust the YAML configs to explore different look-back windows or architectures, then rerun steps 3–4. The autotuning scripts described in 回归实验1109 are intentionally deferred.
