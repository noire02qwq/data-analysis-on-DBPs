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
  - 不再保存固定 checkpoint，而是每个 epoch 记录 train/val MSE；只有当 train loss 下降到初始值的 1/3 时才开始更新 best。Patience 计数从「train loss ≤ 1/3」或「第 31 轮」(先到者)开始，之后若 train/val MSE 在 50 轮内都没有刷新且总轮数 ≥ 80 就提前停止；若训练到第 100 轮仍达不到 1/3 阈值则强制终止。  
  - PyTorch models output `best_model.pt` / `last_model.pt` plus one `training_curve.png` + `loss_history.csv` (均为 MSE)。XGBoost 依次训练 8 个单输出 booster，每个目标单独产出 `training_curve_<target>.png` / `loss_history_<target>.csv` 及 best/last `*.json`。  
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

## 5. Autotuning (LSTM / RNN)
- **Purpose:** Sweep the grids specified in `models/configs/lstm_grid.yaml` / `models/configs/rnn_grid.yaml` by repeatedly launching the training script with different hyperparameters and logging the best validation loss.
- **Commands:**  
  ```bash
  python scripts/autotune_lstm.py [--max-trials N --start-index K]
  python scripts/autotune_rnn.py  [--max-trials N --start-index K]
  ```  
  Use `nohup ... &` if you plan to keep them running overnight.
- **Behaviour:** For each grid point the script creates `scripts/outputs/<model>/<run_id>/config.yaml`, invokes `train_regression.py`, and appends a row to `scripts/outputs/<model>/autotune_results.csv` containing the run id, hyperparameters, and the recorded best validation loss. Runs are numbered sequentially, so you can pause/resume without overwriting earlier trials.

## 6. Rate-Based Regression (`变化率回归1115`)
- **Purpose:** Implements the `prompt-draft/变化率回归1115.md` request by changing every target to `(PPL - RT) / RT` during training, then converts predictions back to absolute values for validation/test metrics.
- **Commands:**  
  ```bash
  python scripts/rate_train_regression.py --config models/configs/<config>.yaml
  python scripts/rate_test_regression.py  --model-dir scripts/outputs/rate_<model_name>
  python scripts/rate_autotune_lstm.py    [--max-trials N --start-index K]
  python scripts/rate_autotune_rnn.py     [--max-trials N --start-index K]
  ```  
  The YAML configs are identical to the direct-regression runs; the scripts automatically prefix the output folders with `rate_`.
- **Key differences vs. the direct pipeline:**  
  - Targets are rates during optimization, but validation/test losses are computed on the recovered absolute values so that MSE stays in the original units.  
  - `metadata.json` now records `target_mode: "rate"` plus the RT column paired with each target, allowing `rate_test_regression.py` to rebuild the conversion without extra inputs.

## 7. Autotuning MLP / MLP_WITH_HISTORY
- **Purpose:** Explore the dense architectures described in `prompt-draft/调参脚本mlp-gbdt1116.md` by sweeping the number/width of middle layers plus optimization hyperparameters. The “hidden layers” are defined via `{mid_layer_count, mid_layer_size}` so the actual list becomes `[2×size] + [size]×count + [size/2]`.
- **Commands:**  
  ```bash
  python scripts/autotune_mlp.py --base-config models/configs/mlp_config.yaml \
       --grid-config models/configs/mlp_grid.yaml [--max-trials N --start-index K]

  python scripts/autotune_mlp_history.py --base-config models/configs/mlp_with_history_config.yaml \
       --grid-config models/configs/mlp_with_history_grid.yaml [--max-trials N --start-index K]
  ```
  Use the `rate_` counterparts (same flags) if you want the `(PPL-RT)/RT` objective.
- **Behaviour:** The shared runner writes numbered folders under `scripts/outputs/<model>/`, copies the effective config, and appends `{mid_layer_count, mid_layer_size, history_length, dropout, batch_size, learning_rate, weight_decay}` + the resulting validation MSE to `autotune_results.csv`. Use `--max-trials` / `--start-index` to split long sweeps, or `nohup ... &` to keep them running overnight.

## 8. Gradient-Boosted Trees (XGBoost / LightGBM / CatBoost)
- **Purpose:** Cover the expanded hyperparameter ranges requested in `prompt-draft/调参脚本mlp-gbdt1116.md`, giving the tree-based baselines more capacity.
- **Commands:**
  ```bash
  python scripts/autotune_xgboost.py  --base-config models/configs/xgboost_config.yaml \
       --grid-config models/configs/xgboost_grid.yaml   [--max-trials N --start-index K]
  python scripts/rate_autotune_xgboost.py  --base-config models/configs/xgboost_config.yaml \
       --grid-config models/configs/xgboost_grid.yaml   [--max-trials N --start-index K]

  python scripts/autotune_lightgbm.py --base-config models/configs/lightgbm_config.yaml \
       --grid-config models/configs/lightgbm_grid.yaml [--max-trials N --start-index K]
  python scripts/rate_autotune_lightgbm.py --base-config models/configs/lightgbm_config.yaml \
       --grid-config models/configs/lightgbm_grid.yaml [--max-trials N --start-index K]

  python scripts/autotune_catboost.py  --base-config models/configs/catboost_config.yaml \
       --grid-config models/configs/catboost_grid.yaml [--max-trials N --start-index K]
  python scripts/rate_autotune_catboost.py  --base-config models/configs/catboost_config.yaml \
       --grid-config models/configs/catboost_grid.yaml [--max-trials N --start-index K]
  ```
- **Behaviour:** Same hill-climbing loop as the neural nets: each run writes a temporary folder under `scripts/outputs/<model_name>/`, invokes the corresponding training script (`train_regression.py` or `rate_train_regression.py` for the rate-aware XGBoost), and logs the selected hyperparameters plus the primary validation loss (TRC-PPL1 in the case of the tree ensembles) into `autotune_results.csv`.
