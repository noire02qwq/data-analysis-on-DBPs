#!/usr/bin/env python3
"""
Train final models with best parameters from Bayesian optimization.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import tomli
import tomli_w

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train final models with best parameters from Bayesian optimization"
    )
    parser.add_argument(
        "--bayes-dir",
        default="outputs/final_all_models",
        help="Directory containing Bayesian optimization results",
    )
    parser.add_argument(
        "--base-config-dir",
        default="models/configs",
        help="Directory containing base configs",
    )
    return parser.parse_args()


def find_best_params(results_csv: Path) -> Dict:
    """Find best parameters from Bayesian optimization results."""
    df = pd.read_csv(results_csv)
    # Find row with minimum value
    best_row = df.loc[df["value"].idxmin()]
    # Extract all parameters
    params = {}
    for col in df.columns:
        if col not in ["trial_number", "value", "state"]:
            if not pd.isna(best_row[col]):
                # Try to parse as proper type
                try:
                    if "." in str(best_row[col]):
                        params[col] = float(best_row[col])
                    else:
                        params[col] = int(best_row[col])
                except:
                    params[col] = best_row[col]
    return params


def merge_params_into_config(base_config: Dict, params: Dict) -> Dict:
    """Merge best parameters into base config."""
    config = base_config.copy()

    for param_key, param_value in params.items():
        # Format is like "parameters.history_length" → (section, param)
        if "." in param_key:
            section_name, param_name = param_key.split(".", 1)
            if section_name == "parameters":
                # Parameters go to either model or training section
                model_params = [
                    "history_length", "units", "num_layers", "dropout", "max_depth",
                    "subsample", "colsample_bytree", "gamma", "reg_lambda", "min_child_weight",
                    "min_child_samples", "num_leaves", "reg_alpha", "depth", "l2_leaf_reg",
                    "random_strength", "bagging_temperature", "d_model", "nhead", "num_encoder_layers",
                    "dim_feedforward", "d_model", "n_layers", "d_state", "d_conv", "expand"
                ]
                training_params = ["batch_size", "learning_rate", "weight_decay", "max_epochs"]

                if param_name in model_params:
                    if "model" not in config:
                        config["model"] = {}
                    config["model"][param_name] = param_value
                elif param_name in training_params:
                    if "training" not in config:
                        config["training"] = {}
                    config["training"][param_name] = param_value

    return config


def train_final_model(
    model_name: str,
    model_type: str,
    base_config_path: Path,
    results_csv: Path,
    output_dir: Path,
) -> bool:
    """Train final model with best parameters."""
    print(f"\n{model_name.upper()}:")
    print(f"  Loading Bayesian results from: {results_csv}")

    # Find best parameters
    if not results_csv.exists():
        print(f"  ✗ Results file not found: {results_csv}")
        return False

    best_params = find_best_params(results_csv)
    print(f"  Best validation loss: {best_params.get('value', 'N/A')}")

    # Load base config
    with open(base_config_path, "rb") as f:
        base_config = tomli.load(f)

    # Merge best parameters
    final_config = merge_params_into_config(base_config, best_params)

    # Update output paths to match the original dataset
    if "data" in final_config:
        final_config["data"]["train_csv"] = "data/train.csv"
        final_config["data"]["val_csv"] = "data/val.csv"
        final_config["data"]["test_csv"] = "data/test.csv"
        # Input all DT and RT columns
        final_config["data"]["input_columns"] = [
            "minutes_since_start",
            "TRC-DT", "pH-DT", "cond-DT", "TRC-RT", "pH-RT", "fDOM-RT", "DO-RT",
            "TOC-RT", "DOC-RT",
        ]
        # Output PPL1 and PPL2 for all parameters (TRC, pH, cond, TOC, DOC)
        final_config["data"]["output_columns"] = [
            "TRC-PPL1", "TRC-PPL2", "pH-PPL1", "pH-PPL2",
            "cond-PPL1", "cond-PPL2", "TOC-PPL1", "TOC-PPL2",
            "DOC-PPL1", "DOC-PPL2",
        ]

    # Write final config
    final_config_path = output_dir / f"{model_name}_final_config.toml"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(final_config_path, "wb") as f:
        tomli_w.dump(final_config, f)

    print(f"  Final config written to: {final_config_path}")
    print(f"  Training final model...")

    # Run training
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train.py"),
        "--config", str(final_config_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            cwd=str(REPO_ROOT),
        )

        if result.returncode != 0:
            print(f"  ✗ Training failed: {result.stderr[:200]}")
            return False

        print(f"  ✓ Training completed")
        return True

    except subprocess.TimeoutExpired:
        print(f"  ✗ Training timed out after 1 hour")
        return False
    except Exception as e:
        print(f"  ✗ Training error: {e}")
        return False


def main():
    args = parse_args()
    bayes_dir = Path(args.bayes_dir)
    base_config_dir = Path(args.base_config_dir)

    models = [
        {
            "name": "xgboost",
            "type": "XGBOOST",
            "base_config": base_config_dir / "xgboost_config.toml",
            "bayes_results": bayes_dir / "xgboost" / "bayes_optimization_results.csv",
        },
        {
            "name": "lightgbm",
            "type": "LIGHTGBM",
            "base_config": base_config_dir / "lightgbm_config.toml",
            "bayes_results": bayes_dir / "lightgbm" / "bayes_optimization_results.csv",
        },
        {
            "name": "catboost",
            "type": "CATBOOST",
            "base_config": base_config_dir / "catboost_config.toml",
            "bayes_results": bayes_dir / "catboost" / "bayes_optimization_results.csv",
        },
        {
            "name": "rnn",
            "type": "RNN",
            "base_config": base_config_dir / "rnn_config.toml",
            "bayes_results": bayes_dir / "rnn" / "bayes_optimization_results.csv",
        },
    ]

    print("="*80)
    print("TRAIN FINAL MODELS WITH BEST PARAMETERS")
    print("="*80)

    success_count = 0
    results = {}

    for model in models:
        success = train_final_model(
            model["name"],
            model["type"],
            model["base_config"],
            model["bayes_results"],
            bayes_dir / model["name"] / "final",
        )
        results[model["name"]] = "success" if success else "failed"
        if success:
            success_count += 1

    print(f"\n{'='*80}")
    print(f"COMPLETE: {success_count}/{len(models)} models trained successfully")
    print(f"{'='*80}")

    # Save results
    with open(bayes_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {bayes_dir / 'training_results.json'}")


if __name__ == "__main__":
    main()