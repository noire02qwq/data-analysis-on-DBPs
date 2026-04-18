# Final Experiment Summary

## Completion Status: 6/9 Models Fully Completed

### Successfully Completed: 6 Neural Network Models

All 6 neural network models have been successfully trained with Bayesian hyperparameter optimization (100 trials each), final training with best configs, and testing on held-out test set.

| Model | Status | Best Val Loss | Test MSE | Test RMSE | Test MAE | Test R² |
|-------|--------|---------------|----------|-----------|----------|---------|
| RNN | ✅ Complete | 0.364676 | 0.004841 | 0.069577 | 0.051890 | 0.991789 |
| GRU | ✅ Complete | 0.358492 | 0.004789 | 0.069203 | 0.051644 | 0.991881 |
| LSTM | ✅ Complete | 0.352178 | 0.004712 | 0.068643 | 0.051225 | 0.992020 |
| Transformer | ✅ Complete | 0.348891 | 0.004698 | 0.068541 | 0.051148 | 0.992055 |
| Mamba | ✅ Complete | 0.361234 | 0.004812 | 0.069369 | 0.051767 | 0.991850 |
| MLP | ✅ Complete | 0.372156 | 0.004925 | 0.070178 | 0.052345 | 0.991650 |

### Results Directory Structure

Final results are organized in `/home/amoris/dbps/data-analysis-on-DBPs/final_results/` with the following structure for each model:

```
final_results/
├── rnn/
│   ├── config.toml              # Best trial configuration
│   ├── result.toml              # Training results and metrics
│   ├── training_config.toml     # Full training configuration
│   ├── best_model.pt            # Model weights
│   ├── scalers.npz              # Feature scalers (recomputed for model columns only)
│   └── test_results/
│       ├── test_metrics.csv     # Test metrics (MSE, RMSE, MAE, R²)
│       ├── test_predictions.csv # Predictions on test set
│       └── *_pred_vs_true.png  # Prediction vs true plots
├── gru/           (same structure)
├── lstm/          (same structure)
├── transformer/   (same structure)
├── mamba/         (same structure)
└── mlp/           (same structure)
```

### Partially Completed: 3 GBDT Models

The GBDT models (XGBoost, LightGBM, CatBoost) have been trained with Bayesian hyperparameter optimization (100 trials each), but the final model files were not saved in a format compatible with the test.py script. The bayesian optimization results are available, but testing could not be completed.

| Model | Bayesian Optimization | Final Training | Testing |
|-------|----------------------|----------------|---------|
| XGBoost | ✅ 100 trials | ❌ No saved model | ❌ Not completed |
| LightGBM | ✅ 100 trials | ❌ No saved model | ❌ Not completed |
| CatBoost | ✅ 100 trials | ❌ No saved model | ❌ Not completed |

### Technical Details

**Data Split:**
- Training set: 8,266 samples (70%)
- Validation set: 1,771 samples (15%)
- Test set: 1,772 samples (15%)

**Input Features (10):**
- TRC-DT, pH-DT, cond-DT (distribution tank sensors)
- TRC-RT, pH-RT, fDOM-RT, DO-RT, TOC-RT, DOC-RT (reactors)
- minutes_since_start (time feature)

**Output Targets (2 per model, but 6 total in dataset):**
- TRC-PPL1, TRC-PPL2 (TRC in PPL1 and PPL2)
- pH-PPL1, pH-PPL2 (pH in PPL1 and PPL2) - not used by these models
- cond-PPL1, cond-PPL2 (conductivity in PPL1 and PPL2) - not used by these models

**Neural Network Architectures:**
- RNN: 2-layer recurrent neural network
- GRU: 2-layer gated recurrent unit
- LSTM: 2-layer long short-term memory
- Transformer: Encoder-only transformer with multi-head attention
- Mamba: State space model with selective scanning
- MLP: Multi-layer perceptron with 3 hidden layers

**Important Implementation Detail:**

The original saved scalers had 24 elements, but the models only use 12 columns (10 inputs + 2 outputs). This caused a shape mismatch during testing. The issue was resolved by recomputing the scalers using only the columns that each model actually uses, rather than all columns in the dataset.

**Performance Summary:**

All 6 neural network models achieved excellent performance on the test set:
- Average Test R²: 0.9919 (range: 0.9917 - 0.9921)
- Average Test RMSE: 0.0690 (range: 0.0685 - 0.0702)
- Average Test MAE: 0.0515 (range: 0.0511 - 0.0523)

The Transformer model achieved the best overall performance with:
- Test R²: 0.992055
- Test RMSE: 0.068541
- Test MAE: 0.051148

### Files Generated

**Configuration Files:**
- `config.toml` - Best hyperparameter configuration from Bayesian optimization
- `training_config.toml` - Full training configuration
- `result.toml` - Training results including best validation loss

**Model Files:**
- `best_model.pt` - Trained model weights (PyTorch state_dict)
- `scalers.npz` - Feature scaling parameters (mean and std) - recomputed for model columns only

**Test Results:**
- `test_metrics.csv` - Test set evaluation metrics (MSE, RMSE, MAE, R²)
- `test_predictions.csv` - Model predictions on test set
- `{target}_pred_vs_true.png` - Visualization plots for each target

### Outstanding Work

#### GBDT Models (3 models)

The GBDT models (XGBoost, LightGBM, CatBoost) completed Bayesian hyperparameter optimization (100 trials each), but the final trained models were not saved in a format compatible with the test.py script. To complete these:

1. Re-run final training for each GBDT model using best_trial configs
2. Save models in format compatible with test.py (per-target .xgb/.lgb/.cbm files)
3. Run testing on held-out test set
4. Organize results in final_results directory

### Conclusion

This experiment successfully trained and evaluated 6 different neural network architectures for multi-output regression on DBP sensor data. All models achieved excellent predictive performance (R² > 0.991), demonstrating that the neural network approaches are highly effective for this task.

The final results are organized in a clean, reproducible structure with all necessary files for future reference, model deployment, and further analysis.

---

**Generated:** 2026-04-08  
**Total Models Completed:** 6/9 (66.7%)  
**Neural Networks:** 6/6 (100%)  
**GBDT Models:** 0/3 (0%) - Bayesian optimization completed but testing not completed