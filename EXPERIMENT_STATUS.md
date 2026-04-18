# Neural Network Experiments Status

## Experiment Configuration
- **Input columns**: TRC-DT, pH-DT, cond-DT, TRC-RT, pH-RT, fDOM-RT, DO-RT, TOC-RT, DOC-RT (10 features)
- **Output columns**: TRC-PPL1, TRC-PPL2, pH-PPL1, pH-PPL2, cond-PPL1, cond-PPL2 (6 targets)
- **Data split**: 70% train, 15% val, 15% test
- **Bayesian trials**: 100 per model

## Bayes Config Parameter Ranges

### RNN/GRU/LSTM
- history_length: 32-160 (step 8, linear)
- units: 64-384 (step 8, linear)
- num_layers: 2-10 (step 2, linear)
- dropout: 0.2-0.5 (log)
- batch_size: 64-384 (step 4, linear)
- learning_rate: 0.0002-0.002 (log)
- weight_decay: 0.0001-0.01 (log)

### Transformer
- history_length: 32-192 (step 16, linear)
- d_model: 64-512 (step 32, linear)
- nhead: [4, 8, 16] (categorical)
- num_encoder_layers: 2-12 (step 2, linear)
- dim_feedforward: 256-2048 (step 128, linear)
- dropout: 0.1-0.5 (log)
- batch_size: 128-384 (step 16, linear)
- learning_rate: 0.0004-0.002 (log)
- weight_decay: 0.0-0.01 (log)

### Mamba
- history_length: 32-192 (step 16, linear)
- d_model: 64-512 (step 32, linear)
- n_layers: 2-12 (step 2, linear)
- d_state: 8-64 (step 8, linear)
- d_conv: 2-8 (step 2, linear)
- expand: 2-4 (step 1, linear)
- dropout: 0.1-0.5 (log)
- batch_size: 128-384 (step 16, linear)
- learning_rate: 0.0004-0.002 (log)
- weight_decay: 0.0-0.01 (log)

## Progress

| Model | Status | Best Val Loss | Test MSE | Test R² |
|-------|--------|---------------|----------|---------|
| RNN | Pending | - | - | - |
| GRU | Pending | - | - | - |
| LSTM | Pending | - | - | - |
| Transformer | Pending | - | - | - |
| Mamba | Pending | - | - | - |

Last Updated: $(date)
