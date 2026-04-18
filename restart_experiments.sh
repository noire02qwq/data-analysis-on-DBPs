#!/bin/bash
# Restart LSTM, Transformer, Mamba experiments

cd /home/amoris/dbps/data-analysis-on-DBPs

# Create output directories
mkdir -p outputs/nn_lstm_final_v2
mkdir -p outputs/nn_transformer_final_v2
mkdir -p outputs/nn_mamba_final_v2

# Start LSTM experiment
conda run -n torch python scripts/autotune.py \
    --model-type LSTM \
    --base-config models/configs/lstm_config.toml \
    --bayes-config models/configs/lstm_bayes.toml \
    --n-trials 100 \
    --output-dir outputs/nn_lstm_final_v2 \
    > outputs/nn_lstm_final_v2/experiment.log 2>&1 &
LSTM_PID=$!
echo "LSTM started with PID $LSTM_PID"

# Start Transformer experiment
conda run -n torch python scripts/autotune.py \
    --model-type TRANSFORMER \
    --base-config models/configs/transformer_config.toml \
    --bayes-config models/configs/transformer_bayes.toml \
    --n-trials 100 \
    --output-dir outputs/nn_transformer_final_v2 \
    > outputs/nn_transformer_final_v2/experiment.log 2>&1 &
TRANS_PID=$!
echo "Transformer started with PID $TRANS_PID"

# Start Mamba experiment
conda run -n torch python scripts/autotune.py \
    --model-type MAMBA \
    --base-config models/configs/mamba_config.toml \
    --bayes-config models/configs/mamba_bayes.toml \
    --n-trials 100 \
    --output-dir outputs/nn_mamba_final_v2 \
    > outputs/nn_mamba_final_v2/experiment.log 2>&1 &
MAMBA_PID=$!
echo "Mamba started with PID $MAMBA_PID"

echo ""
echo "All experiments started!"
echo "Monitor with: tail -f outputs/nn_*/experiment.log"
