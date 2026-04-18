#!/usr/bin/env python3
"""
Check status of all 9 models:
- XGBoost, LightGBM, CatBoost (GBDT)
- MLP, RNN, GRU, LSTM, Transformer, Mamba (NN)

Check for:
1. Bayesian optimization results
2. Final trained models
3. Test results
4. Visualizations
"""

from __future__ import annotations

import argparse
import json
import tomli
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check status of all 9 models"
    )
    parser.add_argument(
        "--output-json",
        help="Output JSON file for status report",
    )
    return parser.parse_args()


def check_model_directory(model_name: str) -> Dict:
    """Check a specific model directory for completion status."""
    status = {
        "model": model_name,
        "has_bayes_opt": False,
        "has_final_model": False,
        "has_test_results": False,
        "has_visualizations": False,
        "best_val_loss": None,
        "test_loss": None,
        "directories": [],
    }

    # Look for model directories
    model_dirs = []
    for item in OUTPUTS_DIR.iterdir():
        if not item.is_dir():
            continue
        dir_name = item.name.lower()
        if model_name.lower() in dir_name:
            model_dirs.append(item)

    status["directories"] = [str(d.relative_to(OUTPUTS_DIR)) for d in model_dirs]

    if not model_dirs:
        return status

    # Check each directory
    for model_dir in model_dirs:
        # Check for bayesian optimization results
        bayes_files = list(model_dir.glob("*bayes*")) + list(model_dir.glob("*autotune*"))
        if bayes_files:
            status["has_bayes_opt"] = True

        # Check for final model files
        model_files = list(model_dir.glob("*.pt")) + list(model_dir.glob("*.xgb")) + \
                      list(model_dir.glob("*.lgb")) + list(model_dir.glob("*.cbm"))
        if model_files:
            status["has_final_model"] = True

        # Check for result.toml files
        result_files = list(model_dir.glob("**/result.toml"))
        for result_file in result_files:
            try:
                with open(result_file, "rb") as f:
                    result = tomli.load(f)
                if "eval" in result:
                    eval_data = result["eval"]
                    if "best_val_loss" in eval_data:
                        status["best_val_loss"] = eval_data["best_val_loss"]
                    if "test_loss" in eval_data:
                        status["test_loss"] = eval_data["test_loss"]
            except:
                pass

        # Check for visualizations
        viz_files = list(model_dir.glob("**/*.png")) + list(model_dir.glob("**/*.jpg")) + \
                    list(model_dir.glob("**/*.svg"))
        if viz_files:
            status["has_visualizations"] = True

        # Check for test results
        test_files = list(model_dir.glob("**/*test*")) + list(model_dir.glob("**/*predict*"))
        if test_files:
            status["has_test_results"] = True

    return status


def check_comprehensive_experiment() -> Dict:
    """Check comprehensive experiment directory."""
    comp_dir = OUTPUTS_DIR / "comprehensive_experiment"
    status = {
        "exists": comp_dir.exists(),
        "models_completed": [],
        "summary": None,
    }

    if not comp_dir.exists():
        return status

    # Check for summary file
    summary_file = comp_dir / "experiment_summary.json"
    if summary_file.exists():
        try:
            with open(summary_file) as f:
                status["summary"] = json.load(f)
        except:
            pass

    # Check which models have been completed
    models_dir = comp_dir / "models"
    if models_dir.exists():
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                completion_file = model_dir / "experiment_complete.txt"
                if completion_file.exists():
                    status["models_completed"].append(model_dir.name)

    return status


def main():
    args = parse_args()

    models_to_check = [
        "xgboost", "lightgbm", "catboost",
        "mlp", "rnn", "gru", "lstm",
        "transformer", "mamba"
    ]

    print("="*80)
    print("MODEL STATUS CHECK")
    print("="*80)

    all_status = {}
    completed_count = 0
    partially_completed = 0
    not_started = 0

    for model_name in models_to_check:
        print(f"\n{model_name.upper()}:")
        status = check_model_directory(model_name)
        all_status[model_name] = status

        # Determine completion level
        criteria_met = sum([
            status["has_final_model"],
            status["has_test_results"],
            status["best_val_loss"] is not None,
        ])

        if criteria_met >= 3:
            print(f"  ✓ Complete (val_loss: {status.get('best_val_loss', 'N/A')})")
            completed_count += 1
        elif criteria_met >= 1:
            print(f"  ~ Partially complete")
            partially_completed += 1
        else:
            print(f"  ✗ Not started")
            not_started += 1

        if status["directories"]:
            print(f"  Directories: {', '.join(status['directories'][:3])}")
            if len(status["directories"]) > 3:
                print(f"    ... and {len(status['directories']) - 3} more")

    # Check comprehensive experiment
    print(f"\n{'='*80}")
    print("COMPREHENSIVE EXPERIMENT")
    print(f"{'='*80}")
    comp_status = check_comprehensive_experiment()
    if comp_status["exists"]:
        print(f"  ✓ Comprehensive experiment exists")
        if comp_status["summary"]:
            summary = comp_status["summary"]
            print(f"    Models run: {summary.get('models_run', [])}")
            print(f"    Results: {summary.get('results', {})}")
        if comp_status["models_completed"]:
            print(f"    Models completed: {comp_status['models_completed']}")
    else:
        print(f"  ✗ No comprehensive experiment found")

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total models: {len(models_to_check)}")
    print(f"  Completed: {completed_count}")
    print(f"  Partially complete: {partially_completed}")
    print(f"  Not started: {not_started}")
    print(f"{'='*80}")

    # Save to JSON if requested
    if args.output_json:
        output_data = {
            "models": all_status,
            "comprehensive_experiment": comp_status,
            "summary": {
                "total": len(models_to_check),
                "completed": completed_count,
                "partial": partially_completed,
                "not_started": not_started,
            }
        }
        with open(args.output_json, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nStatus saved to: {args.output_json}")


if __name__ == "__main__":
    main()