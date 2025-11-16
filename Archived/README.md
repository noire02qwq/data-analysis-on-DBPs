# Archived Results

This folder snapshots the best run from each major autotuning session. Every subdirectory follows the convention `<run_id><model>`; inside you will find the exact artifacts copied from `scripts/outputs/<model_name>/<run_id>/`, including:

- `metadata.json`: captured configuration, split info, best/last checkpoints, and validation metrics.
- `scalers.npz`: mean/std used for normalization.
- Model checkpoints (PyTorch `.pt`, XGBoost `.json`, LightGBM `.txt`, CatBoost `.cbm`).
- Training curves, loss history CSVs, and prediction plots.

Use these archives as immutable references when comparing new experiments or reproducing best-known settings. To rerun evaluation on an archived model, point `scripts/test_regression.py` or `scripts/rate_test_regression.py` at the archived folder just as you would the original `scripts/outputs/...` path.
