# Final Summary - DBPs Regression Pipeline

## Bayesian Optimization Bug Fix

### Problem Identified
The `autotune.py` script was finding stale result files from previous runs instead of the current trial's actual training output. This caused all trials to return identical loss values (all showing "Best is trial 0"), making the optimization meaningless.

### Root Cause
The result file discovery logic in `scripts/autotune.py` (lines 183-237) was not filtering by time. It would find ANY recent result file, including those from previous runs.

### Solution Implemented
Added time-based filtering using file modification time (`mtime`) to ensure only results created AFTER the trial started are considered valid:

```python
# Added time tracking before training
trial_start_time = time.time()

# Added time-based filtering
if mtime < trial_start_time - 30:  # 30 second buffer
    continue
```

Also fixed a secondary null pointer bug where `result_file` could be `None` when calling `.exists()`.

### Transformer Positional Encoding Bug Fix
Fixed a dimension mismatch bug in `models/transformer_regressor.py` where odd `d_model` values caused the positional encoding to fail. Added proper handling for odd dimensions in the cosine term assignment.

## Results After Fix

### Bayesian Optimization Trials
- **Transformer**: 56 completed trials (diverse loss values confirmed)
- **Mamba**: 200 completed trials (diverse loss values confirmed)
- **Other 7 models**: 200 completed trials each

### 9-Model Performance Summary (Sorted by Validation Loss)

| Rank | Model      | Val Loss   | Test Loss  | Best Epoch |
|------|------------|------------|------------|------------|
| 1    | XGBoost    | 0.325805   | 1.856729   | N/A        |
| 2    | RNN        | 0.333258   | 1.246486   | 8          |
| 3    | LightGBM   | 0.341931   | 1.852873   | N/A        |
| 4    | CatBoost   | 0.342651   | 1.842369   | N/A        |
| 5    | Mamba      | 0.364888   | 1.350005   | 15         |
| 6    | GRU        | 0.378721   | 1.367175   | 5          |
| 7    | LSTM       | 0.380991   | 1.230882   | 18         |
| 8    | MLP        | 0.388436   | 1.309820   | 7          |
| 9    | Transformer| 0.520805   | 2.164343   | 27         |

### Key Findings

#### Best Models by Metric
- **Best Validation Loss**: XGBoost (0.3258)
- **Best Test Loss**: LSTM (1.2309)
- **Best Generalization** (lowest test/val ratio): RNN (3.7x)
- **Fastest Convergence**: GRU (epoch 5)

#### Model Categories

**Decision Tree Models (GBDT)**
- XGBoost, LightGBM, CatBoost
- Similar validation performance (0.3258 - 0.3427)
- Higher test loss (~1.85) suggesting overfitting to training distribution

**Recurrent Neural Networks**
- RNN, LSTM, GRU
- Good generalization (test loss 1.23-1.37)
- Fast convergence (5-18 epochs)

**Advanced Architectures**
- Transformer: Highest loss, overfitting issues
- Mamba: Reasonable but not superior performance
- MLP: Baseline performance as expected

## Output Directory Structure

```
outputs/
├── 9_model_summary_fixed_bayes.md     # This summary
├── FINAL_SUMMARY.md                   # Final completion report
│
├── transformer_bayes_fixed/           # Fixed bayesian opt for Transformer
│   ├── bayes_optimization_results.csv
│   └── trial_*/
│
├── mamba_bayes_fixed/                 # Fixed bayesian opt for Mamba
│   ├── bayes_optimization_results.csv
│   └── trial_*/
│
├── *_bayes_v3/                        # Other models (200 trials each)
│   └── ...
│
├── transformer_regressor/             # Final trained models
│   └── 20260412_161510/
│
├── mamba_regressor/
│   └── 20260412_162012/
│
└── [other model directories] ...
```

## Files Modified

1. **scripts/autotune.py** (lines 183-237)
   - Added time-based filtering for result file discovery
   - Fixed null pointer bug

2. **models/transformer_regressor.py** (PositionalEncoding class)
   - Fixed dimension mismatch for odd d_model values

## Conclusion

The Bayesian optimization bug has been successfully fixed. Both Transformer and Mamba models now show diverse loss values across trials, confirming that the optimization is working correctly.

The 9-model comparison shows that:
1. **GBDT models** (XGBoost, LightGBM, CatBoost) achieve the best validation performance
2. **RNN-based models** (RNN, LSTM, GRU) show the best generalization to test data
3. **Advanced architectures** (Transformer, Mamba) did not outperform traditional RNNs on this dataset

---

*Report generated: 2026-04-12*
*Bayesian optimization fix: 2026-04-11*
*Final models trained: 2026-04-12*
