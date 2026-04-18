#!/bin/bash
# 运行所有模型的贝叶斯调参（200次trial）

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch

cd /home/amoris/dbps/data-analysis-on-DBPs

echo "Starting all model Bayesian optimization with 200 trials..."
echo "============================================"

MODELS=(
    "MLP:models/configs/mlp_config_v2.toml:models/configs/mlp_bayes_v2.toml"
    "RNN:models/configs/rnn_config_v2.toml:models/configs/rnn_bayes.toml"
    "GRU:models/configs/gru_config.toml:models/configs/gru_bayes.toml"
    "LSTM:models/configs/lstm_config_v2.toml:models/configs/lstm_bayes.toml"
    "TRANSFORMER:models/configs/transformer_config_v2.toml:models/configs/transformer_bayes.toml"
    "MAMBA:models/configs/mamba_config.toml:models/configs/mamba_bayes.toml"
    "XGBOOST:models/configs/xgboost_config_v2.toml:models/configs/xgboost_bayes.toml"
    "LIGHTGBM:models/configs/lightgbm_config_v2.toml:models/configs/lightgbm_bayes.toml"
    "CATBOOST:models/configs/catboost_config_v2.toml:models/configs/catboost_bayes.toml"
)

for model_spec in "${MODELS[@]}"; do
    IFS=':' read -r model_type base_config bayes_config <<< "$model_spec"

    echo ""
    echo "============================================"
    echo "Running: $model_type"
    echo "============================================"

    python scripts/autotune_complete.py \
        --model-type "$model_type" \
        --base-config "$base_config" \
        --bayes-config "$bayes_config" \
        --n-trials 200

    echo "Completed: $model_type"
done

echo ""
echo "============================================"
echo "All models completed!"
echo "============================================"