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
- **Purpose:** Removes the empty `cond-RT` column and realigns the DT/RT/PPL1/PPL2 measurements so that each row represents the same fluid batch (per `prompt-draft/时间对齐.md`).
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
- **Purpose:** Trains either the MLP or LSTM regression model on the aligned data, saving checkpoints, the best model, training curves, and metadata into `models/outputs/<model_name>/`.
- **Default command:**  
  ```bash
  python scripts/train_regression.py --config models/configs/mlp_config.yaml
  ```
- **Available configs:**  
  - `models/configs/mlp_config.yaml`: feed-forward regressor using a 12-step history window.  
  - `models/configs/lstm_config.yaml`: LSTM regressor using a 24-step history window.  
  You can duplicate and modify these YAML files (history length, hidden sizes, batch size, etc.) and pass the new path via `--config`.
- **What the config controls:** model type/name, history length, hidden size/layers, dropout, training hyperparameters (epochs, batch size, LR, weight decay, checkpoint interval, seed), and data source (CSV path + timestamp column).
- **Outputs per run:**  
  - `models/outputs/<model_name>/best_model.pt` and `last_model.pt`  
  - periodic checkpoints under `models/outputs/<model_name>/checkpoints/`  
  - `scalers.npz`, `metadata.json`, `loss_history.csv`, `training_curve.png`, plus a copy of the YAML config.

## 4. Regression Testing (`test_regression.py`)
- **Purpose:** Loads a trained model directory, evaluates the test split, writes `test_predictions.csv`, and creates one True-vs-Predicted plot per target column.
- **Command template:**  
  ```bash
  python scripts/test_regression.py --model-dir models/outputs/mlp_regressor
  ```
- **Key options:**  
  - `--model-dir`: folder holding `metadata.json`, `best_model.pt`, `scalers.npz`, etc.  
  - `--data`: optional override CSV; otherwise it uses the path recorded in `metadata.json`.
- **Outputs:**  
  - Per-target plots (e.g., `DO-PPL1_pred_vs_true.png`).  
  - `test_predictions.csv` with paired true/predicted values.  
  - Console summary of per-target MSE.

## Typical Workflow
1. `python scripts/fill_missing.py`  
2. `python scripts/align_time.py`  
3. `python scripts/train_regression.py --config models/configs/<your_config>.yaml`  
4. `python scripts/test_regression.py --model-dir models/outputs/<model_name>`

This sequence ensures the regression models always see the cleaned, aligned data and that evaluation uses the same scalers and splits recorded during training. Adjust the YAML configs to explore different look-back windows or network sizes, then rerun steps 3–4.***
