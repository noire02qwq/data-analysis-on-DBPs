# Models Directory

This package contains both the PyTorch architectures (MLP, LSTM, RNN), the gradient-boosted learners (XGBoost, LightGBM, CatBoost), and every YAML configuration used by the training scripts.

## Layout

| Path | Description |
| --- | --- |
| `mlp_regressor.py`, `lstm_regressor.py`, `rnn_regressor.py` | PyTorch modules instantiated by `train_regression.py` / `rate_train_regression.py`. |
| `xgboost_regressor.py`, `lightgbm_regressor.py`, `catboost_regressor.py` | Thin wrappers around the respective libraries for saving/loading per-target boosters. |
| `configs/` | All baseline configs and grid specs referenced by the CLI scripts. Every config shares the same schema described below. |

## YAML Schema

Each config breaks down into three sections:

- `model`: required keys are `type` (`MLP`, `MLP_WITH_HISTORY`, `LSTM`, `RNN`, `XGBOOST`, `LIGHTGBM`, `CATBOOST`) and `name` (used as `scripts/outputs/<name>`). The remaining fields configure hidden layers, history length, or tree hyperparameters depending on the model type.
- `training`: controls max epochs (or estimators for GBDTs), optimizer settings, and seed. Even for ensemble models the `max_epochs` field drives boosting rounds so the scripts can keep a consistent interface.
- `data`: points at the aligned CSV and timestamp column. Most experiments use `data/time_aligned_data.csv`.

## Editing Tips

- When defining MLP hidden layers, you can provide an explicit list (e.g., `[1024, 512, 512, 256]`) or rely on the autotune runner with `mid_layer_count/mid_layer_size` to have it populated automatically.
- For LightGBM and CatBoost configs, set the same ranges as the corresponding grid file so that single-run experiments and autotune jobs stay in sync.
- If you copy a config to create a rate-based variant, keep the same contents—the `rate_` scripts reuse the exact YAML and only change how targets/losses are computed.
