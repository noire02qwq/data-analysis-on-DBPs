#!/usr/bin/env python3
"""
Final check of all 9 models and completion status.
Version 2 - properly searches nested directories.
"""

import json
import tomli
from pathlib import Path
from typing import Dict, List, Tuple, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs"

MODELS = [
    "xgboost", "lightgbm", "catboost",
    "mlp", "rnn", "gru", "lstm",
    "transformer", "mamba"
]


def find_latest_complete(model_name: str) -> Optional[Path]:
    """Find the latest complete directory for a model."""
    all_complete = []
    model_name_lower = model_name.lower()

    # Search all subdirectories recursively
    for item in OUTPUTS_DIR.rglob("*"):
        if not item.is_dir():
            continue

        # Check if directory path contains model name (case-insensitive)
        path_str = str(item).lower()
        if model_name_lower not in path_str:
            continue

        # Check if this directory contains a complete result
        has_config = (item / "config.toml").exists()
        has_result = (item / "result.toml").exists()

        if not (has_config and has_result):
            continue

        # Check for at least one model file
        has_model = False
        model_patterns = ['*.pt', '*.xgb', '*.lgb', '*.cbm']
        for pattern in model_patterns:
            if list(item.glob(pattern)):
                has_model = True
                break

        if has_model:
            all_complete.append((item.stat().st_mtime, item))

    if not all_complete:
        return None

    # Sort by modification time (newest last)
    all_complete.sort(key=lambda x: x[0])
    return all_complete[-1][1]


def check_model(model_name: str) -> Dict:
    """Check completion status of a model."""
    status = {
        "model": model_name,
        "has_config": False,
        "has_model_file": False,
        "has_result": False,
        "has_test": False,
        "best_val_loss": None,
        "test_loss": None,
        "latest_dir": None,
        "is_complete": False,
    }

    latest_dir = find_latest_complete(model_name)
    if latest_dir is None:
        return status

    status["latest_dir"] = str(latest_dir.relative_to(OUTPUTS_DIR))
    status["has_config"] = (latest_dir / "config.toml").exists()

    # Check for model files
    model_patterns = ['*.pt', '*.xgb', '*.lgb', '*.cbm']
    for pattern in model_patterns:
        if list(latest_dir.glob(pattern)):
            status["has_model_file"] = True
            break

    # Check for result file
    result_files = list(latest_dir.glob("result.toml"))
    if result_files:
        status["has_result"] = True
        try:
            with open(result_files[0], "rb") as f:
                result = tomli.load(f)
            if "eval" in result:
                eval_data = result["eval"]
                if "best_val_loss" in eval_data:
                    status["best_val_loss"] = eval_data["best_val_loss"]
                if "test_loss" in eval_data:
                    status["test_loss"] = eval_data["test_loss"]
        except Exception as e:
            print(f"Error reading result: {e}")
            pass

    # Check for visualization (training curve)
    if len(list(latest_dir.glob("*training_curve*.png"))) > 0:
        status["has_test"] = True

    # Determine if complete
    status["is_complete"] = (
        status["has_config"] and
        status["has_model_file"] and
        status["has_result"] and
        status["best_val_loss"] is not None
    )

    return status


def main():
    print("="*80)
    print("FINAL CHECK v2 - ALL 9 MODELS")
    print("="*80)

    results = {}
    complete_count = 0

    for model in MODELS:
        status = check_model(model)
        results[model] = status

        print(f"\n{model.upper()}:")
        if status["is_complete"]:
            print(f"  ✓ COMPLETE (val_loss: {status['best_val_loss']:.6f})")
            if status["test_loss"] is not None:
                print(f"    test_loss: {status['test_loss']:.6f}")
            complete_count += 1
        else:
            print(f"  ✗ INCOMPLETE")
            print(f"    Config: {status['has_config']}")
            print(f"    Model file: {status['has_model_file']}")
            print(f"    Result: {status['has_result']}")
            print(f"    Best val loss: {status['best_val_loss']}")
        if status["latest_dir"]:
            print(f"    Latest dir: {status['latest_dir']}")

    print(f"\n{'='*80}")
    print(f"SUMMARY: {complete_count}/{len(MODELS)} models complete")
    print(f"{'='*80}")

    # Save results
    output_file = REPO_ROOT / "model_status_v2.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    if complete_count == len(MODELS):
        print("\n✓ ALL MODELS COMPLETE!")
    else:
        print(f"\n⚠ {len(MODELS) - complete_count} models need completion.")


if __name__ == "__main__":
    main()
