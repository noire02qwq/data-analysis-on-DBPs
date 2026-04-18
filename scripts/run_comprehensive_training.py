#!/usr/bin/env python3
"""Comprehensive training script for all 9 models with best hyperparameters."""

import json
import shutil
import subprocess
import sys
import tomli
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs" / "caww29_final"

INPUT_COLS = ["TRC-DT", "pH-DT", "cond-DT", "TRC-RT", "pH-RT", "cond-RT", "fDOM-RT", "DO-RT", "TOC-RT", "DOC-RT"]
OUTPUT_COLS = ["TRC-PPL1", "TRC-PPL2", "pH-PPL1", "pH-PPL2", "cond-PPL1", "cond-PPL2", "TOC-PPL1", "TOC-PPL2"]

DATA_CONFIG = {
    "train_csv": "data/train.csv",
    "val_csv": "data/val.csv",
    "test_csv": "data/test.csv",
    "input_columns": INPUT_COLS,
    "output_columns": OUTPUT_COLS
}

MODEL_NAME_MAP = {
    "xgboost": "xgboost_regressor",
    "lightgbm": "lightgbm_regressor",
    "catboost": "catboost_regressor",
    "mlp": "mlp_regressor",
    "rnn": "rnn_regressor",
    "gru": "gru_regressor",
    "lstm": "lstm_regressor",
    "mamba": "mamba_regressor",
    "transformer": "transformer_regressor",
}

# Best hyperparameters from 200-trial optimization
BEST_PARAMS = {
    "xgboost": {"model": {"type": "XGBOOST"}},
    "lightgbm": {"model": {"type": "LIGHTGBM"}},
    "catboost": {"model": {"type": "CATBOOST"}},
    "mlp": {"model": {"type": "MLP"}},
    "rnn": {"model": {"type": "RNN"}},
    "gru": {"model": {"type": "GRU"}},
    "lstm": {"model": {"type": "LSTM"}},
    "mamba": {"model": {"type": "MAMBA"}},
    "transformer": {"model": {"type": "TRANSFORMER"}},
}


def run_training(model_name: str) -> bool:
    """Run training for a single model."""
    model_dir = OUTPUT_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = model_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build full config
    config = {
        "model": {"type": BEST_PARAMS[model_name]["model"]["type"], "name": MODEL_NAME_MAP[model_name]},
        "training": {"max_epochs": 150, "patience": 15, "seed": 42},
        "data": DATA_CONFIG
    }

    # Add model-specific params
    config["model"].update(get_model_specific_params(model_name))

    config_path = run_dir / "config.toml"

    # Write config
    import tomli_w
    with config_path.open("wb") as f:
        tomli_w.dump(config, f)

    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print(f"Config: {config_path}")

    # Run training
    cmd = ["python", "scripts/train.py", "--config", str(config_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)

    if result.returncode != 0:
        print(f"ERROR: {result.stderr[:500]}")
        return False

    # Training saves to outputs/{model_name}/{timestamp}
    # Find the output directory
    model_output_dir = REPO_ROOT / "outputs" / MODEL_NAME_MAP[model_name]
    if not model_output_dir.exists():
        print(f"ERROR: Model output directory not found: {model_output_dir}")
        return False

    # Get the most recent output
    subdirs = [d for d in model_output_dir.iterdir() if d.is_dir()]
    if not subdirs:
        print(f"ERROR: No subdirectories in {model_output_dir}")
        return False

    latest_dir = sorted(subdirs, key=lambda x: x.stat().st_mtime)[-1]

    # Copy results to our run_dir
    for item in latest_dir.iterdir():
        dest = run_dir / item.name
        if item.is_file():
            shutil.copy2(item, dest)
        elif item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)

    # Run test
    test_output = run_dir / "test_results"
    test_output.mkdir(parents=True, exist_ok=True)

    test_cmd = ["python", "scripts/test.py", "--model-dir", str(run_dir), "--output-dir", str(test_output)]
    test_result = subprocess.run(test_cmd, capture_output=True, text=True, cwd=REPO_ROOT)

    if test_result.returncode != 0:
        print(f"Test ERROR: {test_result.stderr[:500]}")
        return False

    print(f"  [OK] {model_name} - train & test complete")
    return True


def get_model_specific_params(model_name: str) -> Dict[str, Any]:
    """Get model-specific hyperparameters."""
    params = {
        "xgboost": {
            "colsample_bytree": 0.80, "gamma": 0.003, "learning_rate": 0.014,
            "max_depth": 4, "min_child_weight": 1, "reg_lambda": 2.17, "subsample": 0.66,
        },
        "lightgbm": {
            "colsample_bytree": 0.81, "learning_rate": 0.006, "max_depth": 4,
            "min_child_samples": 69, "reg_alpha": 0.0016, "reg_lambda": 2.25, "subsample": 0.78,
        },
        "catboost": {
            "bagging_temperature": 0.023, "depth": 8, "l2_leaf_reg": 1.53,
            "learning_rate": 0.0027, "random_strength": 0.257, "subsample": 0.82,
        },
        "mlp": {
            "batch_size": 157, "dropout": 0.149, "mid_layer_count": 4, "mid_layer_size": 340,
            "learning_rate": 0.00073, "weight_decay": 0.005,
        },
        "rnn": {
            "history_length": 157, "num_layers": 7, "units": 81, "dropout": 0.36,
            "batch_size": 354, "learning_rate": 0.0003, "weight_decay": 0.00038,
        },
        "gru": {
            "history_length": 123, "num_layers": 5, "units": 126, "dropout": 0.47,
            "batch_size": 375, "learning_rate": 0.00097, "weight_decay": 0.0012,
        },
        "lstm": {
            "history_length": 102, "num_layers": 1, "units": 234, "dropout": 0.139,
            "batch_size": 363, "learning_rate": 0.0003, "weight_decay": 0.0074,
        },
        "mamba": {
            "history_length": 182, "d_model": 107, "n_layers": 6, "d_state": 62,
            "d_conv": 6, "expand": 3, "dropout": 0.27, "batch_size": 196,
            "learning_rate": 0.00097, "weight_decay": 0.00023,
        },
        "transformer": {
            "history_length": 125, "d_model": 120, "nhead": 4, "num_encoder_layers": 5,
            "dim_feedforward": 498, "dropout": 0.083, "batch_size": 243,
            "learning_rate": 0.00033, "weight_decay": 0.0033,
        },
    }
    return params.get(model_name, {})


def main():
    print("Starting comprehensive training for all 9 models...")
    print(f"Output directory: {OUTPUT_DIR}")

    results = {}
    for model_name in BEST_PARAMS.keys():
        try:
            success = run_training(model_name)
            results[model_name] = "OK" if success else "FAILED"
        except Exception as e:
            import traceback
            results[model_name] = f"ERROR: {e}\n{traceback.format_exc()}"
            print(f"  [ERROR] {model_name}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Training Summary:")
    for model_name, status in results.items():
        print(f"  {model_name}: {status}")

    # Save summary
    summary_path = OUTPUT_DIR / "training_summary.json"
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()