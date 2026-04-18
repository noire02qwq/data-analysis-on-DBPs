#!/bin/bash
# Run all 5 NN experiments in parallel

cd /home/amoris/dbps/data-analysis-on-DBPs

# Start RNN experiment
conda run -n torch python scripts/autotune.py \
    --model-type RNN \
    --base-config models/configs/rnn_config.toml \
    --bayes-config models/configs/rnn_bayes.toml \
    --n-trials 100 \
    --output-dir outputs/nn_rnn_final_v2 \
    > outputs/nn_rnn_final_v2/experiment.log 2>&1 &
echo "RNN started with PID $!"

# Start GRU experiment
conda run -n torch python scripts/autotune.py \
    --model-type GRU \
    --base-config models/configs/gru_config.toml \
    --bayes-config models/configs/gru_bayes.toml \
    --n-trials 100 \
    --output-dir outputs/nn_gru_final_v2 \
    > outputs/nn_gru_final_v2/experiment.log 2>&1 &
echo "GRU started with PID $!"

# Start LSTM experiment
conda run -n torch python scripts/autotune.py \
    --model-type LSTM \
    --base-config models/configs/lstm_config.toml \
    --bayes-config models/configs/lstm_bayes.toml \
    --n-trials 100 \
    --output-dir outputs/nn_lstm_final_v2 \
    > outputs/nn_lstm_final_v2/experiment.log 2>&1 &
echo "LSTM started with PID $!"

# Start Transformer experiment
conda run -n torch python scripts/autotune.py \
    --model-type TRANSFORMER \
    --base-config models/configs/transformer_config.toml \
    --bayes-config models/configs/transformer_bayes.toml \
    --n-trials 100 \
    --output-dir outputs/nn_transformer_final_v2 \
    > outputs/nn_transformer_final_v2/experiment.log 2>&1 &
echo "Transformer started with PID $!"

# Start Mamba experiment
conda run -n torch python scripts/autotune.py \
    --model-type MAMBA \
    --base-config models/configs/mamba_config.toml \
    --bayes-config models/configs/mamba_bayes.toml \
    --n-trials 100 \
    --output-dir outputs/nn_mamba_final_v2 \
    > outputs/nn_mamba_final_v2/experiment.log 2>&1 &
echo "Mamba started with PID $!"

echo ""
echo "All 5 experiments started!"
echo "Monitor with: tail -f outputs/nn_*/experiment.log"
