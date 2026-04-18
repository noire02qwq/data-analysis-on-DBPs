# GBDT Experiment Implementation Status

## Completed Features ✓

### 1. Individual Plots Per Output Variable
- Each of the 10 output variables (TRC-PPL1, TRC-PPL2, pH-PPL1, pH-PPL2, cond-PPL1, cond-PPL2, TOC-PPL1, TOC-PPL2, DOC-PPL1, DOC-PPL2) has:
  - Individual prediction vs true plot (saved in `predictions/` directory)
  - Individual y=x scatter plot with R² (saved in `scatter/` directory)

### 2. Loss Curves Per Epoch
- For the best trial, training and validation loss curves over epochs are generated
- Requires saving epoch-by-epoch loss history for each trial
- Saved as `loss_curves_per_epoch.png` in the model directory

### 3. Save Model Parameters
- Trained models are saved to disk using `joblib`
- Saved as `{model_name}_model.pkl` in the model directory

### 4. Save Trial Loss History
- Each trial's epoch-by-epoch loss history is saved
- Saved as `best_trial_epoch_losses.json` for the best trial
- All trials' history saved in `trial_history.json`

## Script Location
- Main script: `gbdt_experiment_final.py`
- Run with: `python gbdt_experiment_final.py --n-trials 100`

## Output Structure
```
outputs/gbdt_experiment/
├── data_split/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── models/
│   ├── xgboost/
│   │   ├── best_config.toml
│   │   ├── model.pkl
│   │   ├── predictions/
│   │   │   ├── TRC-PPL1_prediction.png
│   │   │   └── ... (10 individual plots)
│   │   ├── scatter/
│   │   │   ├── TRC-PPL1_scatter.png
│   │   │   └── ... (10 individual plots)
│   │   ├── loss_curves_per_epoch.png
│   │   ├── best_trial_epoch_losses.json
│   │   └── trial_history.json
│   ├── lightgbm/
│   └── catboost/
└── experiment_summary.json
```
