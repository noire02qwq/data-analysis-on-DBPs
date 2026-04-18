#!/usr/bin/env python3
"""
运行所有模型的贝叶斯调参
- 神经网络模型: MLP, RNN, GRU, LSTM, Transformer, Mamba
  输入: DT和RT的所有列
  输出: PPL1和PPL2的TRC, pH, cond, TOC多输出
- 决策树模型: XGBoost, LightGBM, CatBoost
  输入: DT和RT的所有列
  输出: 只单输出PPL2的TRC
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

# 神经网络模型配置
NN_MODELS = [
    {
        "name": "mlp",
        "model_type": "MLP",
        "base_config": "models/configs/mlp_config_v2.toml",
        "bayes_config": "models/configs/mlp_bayes_v2.toml",
    },
    {
        "name": "rnn",
        "model_type": "RNN",
        "base_config": "models/configs/rnn_config_v2.toml",
        "bayes_config": "models/configs/rnn_bayes.toml",
    },
    {
        "name": "gru",
        "model_type": "GRU",
        "base_config": "models/configs/gru_config.toml",
        "bayes_config": "models/configs/gru_bayes.toml",
    },
    {
        "name": "lstm",
        "model_type": "LSTM",
        "base_config": "models/configs/lstm_config_v2.toml",
        "bayes_config": "models/configs/lstm_bayes.toml",
    },
    {
        "name": "transformer",
        "model_type": "TRANSFORMER",
        "base_config": "models/configs/transformer_config_v2.toml",
        "bayes_config": "models/configs/transformer_bayes.toml",
    },
    {
        "name": "mamba",
        "model_type": "MAMBA",
        "base_config": "models/configs/mamba_config.toml",
        "bayes_config": "models/configs/mamba_bayes.toml",
    },
]

# 决策树模型配置 (单输出TRC-PPL2)
GBDT_MODELS = [
    {
        "name": "xgboost",
        "model_type": "XGBOOST",
        "base_config": "models/configs/xgboost_config_v2_single.toml",
        "bayes_config": "models/configs/xgboost_bayes.toml",
    },
    {
        "name": "lightgbm",
        "model_type": "LIGHTGBM",
        "base_config": "models/configs/lightgbm_config_v2_single.toml",
        "bayes_config": "models/configs/lightgbm_bayes.toml",
    },
    {
        "name": "catboost",
        "model_type": "CATBOOST",
        "base_config": "models/configs/catboost_config_v2_single.toml",
        "bayes_config": "models/configs/catboost_bayes.toml",
    },
]

N_TRIALS = 200


def run_autotune(model_config, n_trials=N_TRIALS):
    """运行单个模型的贝叶斯调参"""
    name = model_config["name"]
    model_type = model_config["model_type"]
    base_config = model_config["base_config"]
    bayes_config = model_config["bayes_config"]

    print(f"\n{'='*60}")
    print(f"Running Bayesian optimization for {name}")
    print(f"{'='*60}")

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "autotune_complete.py"),
        "--model-type", model_type,
        "--base-config", str(REPO_ROOT / base_config),
        "--bayes-config", str(REPO_ROOT / bayes_config),
        "--n-trials", str(n_trials),
    ]

    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return result.returncode == 0


def main():
    # 先运行神经网络模型
    for model in NN_MODELS:
        success = run_autotune(model)
        if not success:
            print(f"Failed: {model['name']}")

    # 再运行决策树模型
    for model in GBDT_MODELS:
        success = run_autotune(model)
        if not success:
            print(f"Failed: {model['name']}")

    print("\nAll models completed!")


if __name__ == "__main__":
    main()