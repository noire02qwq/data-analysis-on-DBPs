#!/usr/bin/env python3
"""Re-run training with fixed data split that has overlapping distributions."""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import shutil
import tomli_w

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs" / "caww29_final_v2"

INPUT_COLS = ["TRC-DT", "pH-DT", "cond-DT", "TRC-RT", "pH-RT", "cond-RT", "fDOM-RT", "DO-RT", "TOC-RT", "DOC-RT"]
OUTPUT_COLS = ["TRC-PPL1", "TRC-PPL2", "pH-PPL1", "pH-PPL2", "cond-PPL1", "cond-PPL2", "TOC-PPL1", "TOC-PPL2"]

DATA_CONFIG = {
    "train_csv": "data/train.csv",
    "val_csv": "data/val.csv",
    "test_csv": "data/test.csv",
    "input_columns": INPUT_COLS,
    "output_columns": OUTPUT_COLS
}

MODEL_MAP = {
    "xgboost": {"type": "XGBOOST", "name": "xgboost_regressor"},
    "lightgbm": {"type": "LIGHTGBM", "name": "lightgbm_regressor"},
    "catboost": {"type": "CATBOOST", "name": "catboost_regressor"},
    "mlp": {"type": "MLP", "name": "mlp_regressor"},
    "rnn": {"type": "RNN", "name": "rnn_regressor"},
    "gru": {"type": "GRU", "name": "gru_regressor"},
    "lstm": {"type": "LSTM", "name": "lstm_regressor"},
    "mamba": {"type": "MAMBA", "name": "mamba_regressor"},
    "transformer": {"type": "TRANSFORMER", "name": "transformer_regressor"},
}

PARAMS = {
    "xgboost": {"max_depth": 4, "learning_rate": 0.014, "subsample": 0.66, "colsample_bytree": 0.80, "gamma": 0.003, "reg_lambda": 2.17},
    "lightgbm": {"max_depth": 4, "learning_rate": 0.006, "subsample": 0.78, "colsample_bytree": 0.81, "reg_lambda": 2.25},
    "catboost": {"depth": 8, "learning_rate": 0.0027, "subsample": 0.82, "l2_leaf_reg": 1.53},
    "mlp": {"mid_layer_count": 4, "mid_layer_size": 340, "dropout": 0.149},
    "rnn": {"history_length": 50, "num_layers": 3, "units": 128, "dropout": 0.2},
    "gru": {"history_length": 50, "num_layers": 3, "units": 128, "dropout": 0.2},
    "lstm": {"history_length": 50, "num_layers": 2, "units": 128, "dropout": 0.2},
    "mamba": {"history_length": 50, "d_model": 128, "n_layers": 4, "d_state": 16, "d_conv": 4, "expand": 2},
    "transformer": {"history_length": 50, "d_model": 128, "nhead": 4, "num_encoder_layers": 3, "dim_feedforward": 256},
}

def run_model(model_name):
    model_dir = OUTPUT_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = model_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": {**MODEL_MAP[model_name], **PARAMS[model_name]},
        "training": {"max_epochs": 150, "patience": 15, "seed": 42},
        "data": DATA_CONFIG
    }

    config_path = run_dir / "config.toml"
    with config_path.open("wb") as f:
        tomli_w.dump(config, f)

    print(f"\n{'='*60}")
    print(f"Training {model_name}...")

    result = subprocess.run(
        ["python", "scripts/train.py", "--config", str(config_path)],
        capture_output=True, text=True, cwd=REPO_ROOT
    )

    if result.returncode != 0:
        print(f"ERROR: {result.stderr[:500]}")
        return False

    # Copy results from outputs/{model_name}/{timestamp}
    model_output = REPO_ROOT / "outputs" / MODEL_MAP[model_name]["name"]
    if model_output.exists():
        subdirs = [d for d in model_output.iterdir() if d.is_dir()]
        if subdirs:
            latest = sorted(subdirs, key=lambda x: x.stat().st_mtime)[-1]
            for item in latest.iterdir():
                dest = run_dir / item.name
                if item.is_file():
                    shutil.copy2(item, dest)
                elif item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)

    # Run test
    test_out = run_dir / "test_results"
    test_out.mkdir(parents=True, exist_ok=True)

    test_result = subprocess.run(
        ["python", "scripts/test.py", "--model-dir", str(run_dir), "--output-dir", str(test_out)],
        capture_output=True, text=True, cwd=REPO_ROOT
    )

    if test_result.returncode != 0:
        print(f"Test ERROR: {test_result.stderr[:300]}")
        return False

    print(f"  [OK] {model_name}")
    return True

def main():
    results = {}
    for model_name in MODEL_MAP.keys():
        try:
            results[model_name] = "OK" if run_model(model_name) else "FAILED"
        except Exception as e:
            results[model_name] = f"ERROR: {e}"
            print(f"  [ERROR] {model_name}: {e}")

    print("\n" + "="*60)
    print("Summary:")
    for k, v in results.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()