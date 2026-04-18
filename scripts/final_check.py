#!/usr/bin/env python3
"""
Final check of all 9 models and completion status.
"""

import json
import tomli
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs"

MODELS = [
    "xgboost", "lightgbm", "catboost",
    "mlp", "rnn", "gru", "lstm",
    "transformer", "mamba"
]

def find_latest_result_dir(base_dir: Path) -> Path | None:
    """Find the latest directory containing a complete result."""
    all_candidates = []

    # Check the base directory itself first
    if (base_dir / "config.toml").exists() and (base_dir / "result.toml").exists():
        all_candidates.append(base_dir)

    # Recursively check subdirectories
    for item in base_dir.rglob("*"):
        if item.is_dir():
            if (item / "config.toml").exists() and (item / "result.toml").exists():
                all_candidates.append(item)

    if not all_candidates:
        return None

    # Sort by modification time (newest first)
    all_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return all_candidates[0]


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

    # Find all directories for this model (top-level)
    model_dirs = []
    for item in OUTPUTS_DIR.iterdir():
        if not item.is_dir():
            continue
        if model_name.lower() in item.name.lower():
            model_dirs.append(item)

    if not model_dirs:
        return status

    # Find the latest directory with a complete result
    latest_dir = find_latest_result_dir(model_dirs[0])
    if latest_dir is not None:
        status["latest_dir"] = str(latest_dir.relative_to(OUTPUTS_DIR))

        # Check for required files
        if (latest_dir / "config.toml").exists():
            status["has_config"] = True

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
            except:
                pass

        # Check for test files
        test_patterns = ['*test*.csv', '*predict*.csv', '*test*.png', '*training_curve*.png']
        for pattern in test_patterns:
            if list(latest_dir.glob(pattern)):
                status["has_test"] = True
                break

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
    print("FINAL CHECK - ALL 9 MODELS")
    print("="*80)

    results = {}
    complete_count = 0

    for model in MODELS:
        status = check_model(model)
        results[model] = status

        print(f"\n{model.upper()}:")
        if status["is_complete"]:
            print(f"  ✓ COMPLETE (val_loss: {status['best_val_loss']:.6f})")
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
    output_file = REPO_ROOT / "model_status.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    if complete_count == len(MODELS):
        print("\n✓ ALL MODELS COMPLETE!")
    else:
        print(f"\n⚠ {len(MODELS) - complete_count} models need completion.")

if __name__ == "__main__":
    main()