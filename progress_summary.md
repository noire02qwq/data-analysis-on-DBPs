# Project Refactoring Progress Report

**Date**: 2026-03-14
**Status**: 🟠 In Progress (Phase 4: Training Script Refactoring)

## 🎯 Completed Milestones

### 1. Cleanup & Preparation
- [x] Cleared old log files and temporary artifacts.
- [x] Deleted redundant/obsolete scripts (trc/other/rate variants).
- [x] Removed "change rate" research scripts as per user request.

### 2. Configuration Migration
- [x] Converted all YAML configurations in `models/configs/` to TOML.
- [x] Standardized the schema to include `input_columns` and `output_columns`.
- [x] Converted Bayesian search-space configs to TOML.

### 3. Data Splitting Utility
- [x] Created `scripts/split_data.py`.
- [x] Implemented shuffling support and fixed row splitting for reproducible training/validation/test sets.
- [x] Verified correctly generates `train.csv`, `val.csv`, and `test.csv`.

## 🏗️ Work in Progress

### 4. Training Script Refactoring
- [x] Developed unified `train.py` supporting PyTorch (MLP, RNN, LSTM, GRU, Transformer) and GBDT (XGBoost, LightGBM, CatBoost) models.
- [x] Implemented ONNX export for best PyTorch models.
- [x] Implemented timestamped output subdirectories with `result.toml` summaries.
- [/] **Current Blocker**: Python process was exiting with status 1 silently during imports. Identified a likely corruption in the `xgboost`/`lightgbm` packages within the `torch` conda environment.
- [/] **Action**: Currently running `pip install --force-reinstall xgboost lightgbm catboost` to restore the environment.

## 📋 Next Steps

1. **Verify Training**: Run smoke tests on refined `train.py` once library repair is complete.
2. **Phase 5: Autotune Refactoring**: Create unified `autotune.py` for Bayesian optimization across all model types.
3. **Phase 6: Test Refactoring**: Create unified `test.py` for model evaluation on external datasets.
4. **Phase 7: Backend Server**: Implement FastAPI server to bridge these scripts to the frontend.

## 📂 Current Directory Layout
```text
scripts/
├── split_data.py  # Done
├── train.py       # Refactoring (Testing)
├── utils.py       # Refactored
└── ...
models/
├── configs/       # All .toml
└── ...
outputs/           # Consolidated training results
```
