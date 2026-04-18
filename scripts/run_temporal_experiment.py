#!/usr/bin/env python3
"""
Comprehensive TEMPORAL experiment script for DBPs regression models.

This script maintains TEMPORAL ORDER throughout (NO SHUFFLING) for time series data.
Uses the existing data/train.csv, data/val.csv, data/test.csv that were split temporally.

1. Runs bayesian optimization (200 trials) for each model:
   - XGBoost, LightGBM, CatBoost (GBDT models)
   - MLP, RNN, GRU, LSTM, Mamba, Transformer (NN models)
2. For each model, saves:
   - Best config TOML
   - Results TOML
   - Loss history CSV
   - Predictions vs true values CSV
3. Generates plots with timestamps on x-axis (day-level)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import optuna
import polars as pl
import pandas as pd

try:
    import tomllib as tomli
except ImportError:
    import tomli

try:
    import tomli_w
except ImportError:
    tomli_w = None

from sklearn.metrics import r2_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils import (
    DatasetBundle,
    build_dataset_bundle,
    compute_scalers,
    get_feature_and_target_indices,
    load_time_series,
    scale_values,
)


# Model configurations - using temporal split data
MODEL_CONFIGS = {
    "xgboost": {
        "model_type": "XGBOOST",
        "base_config": "models/configs/xgboost_config.toml",
        "bayes_config": "models/configs/xgboost_bayes.toml",
    },
    "lightgbm": {
        "model_type": "LIGHTGBM",
        "base_config": "models/configs/lightgbm_config.toml",
        "bayes_config": "models/configs/lightgbm_bayes.toml",
    },
    "catboost": {
        "model_type": "CATBOOST",
        "base_config": "models/configs/catboost_config.toml",
        "bayes_config": "models/configs/catboost_bayes.toml",
    },
    "mlp": {
        "model_type": "MLP",
        "base_config": "models/configs/mlp_config.toml",
        "bayes_config": "models/configs/mlp_bayes.toml",
    },
    "rnn": {
        "model_type": "RNN",
        "base_config": "models/configs/rnn_config.toml",
        "bayes_config": "models/configs/rnn_bayes.toml",
    },
    "gru": {
        "model_type": "GRU",
        "base_config": "models/configs/gru_config.toml",
        "bayes_config": "models/configs/gru_bayes.toml",
    },
    "lstm": {
        "model_type": "LSTM",
        "base_config": "models/configs/lstm_config.toml",
        "bayes_config": "models/configs/lstm_bayes.toml",
    },
    "mamba": {
        "model_type": "MAMBA",
        "base_config": "models/configs/mamba_config.toml",
        "bayes_config": "models/configs/mamba_bayes.toml",
    },
    "transformer": {
        "model_type": "TRANSFORMER",
        "base_config": "models/configs/transformer_config.toml",
        "bayes_config": "models/configs/transformer_bayes.toml",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run comprehensive TEMPORAL experiment with all models (NO SHUFFLING)."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/temporal_experiment",
        help="Output directory (default: outputs/temporal_experiment)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=200,
        help="Number of trials for bayesian optimization (default: 200)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CONFIGS.keys()) + ["all"],
        default=["all"],
        help="Models to run (default: all)",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        default=True,
        help="Skip models that have already been completed (default: True)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as fh:
        return tomli.load(fh)


def save_toml(path: Path, data: Dict[str, Any]) -> None:
    with path.open("wb") as fh:
        tomli_w.dump(data, fh)


def run_training(
    config_path: Path,
    model_key: str,
    trial_num: int,
) -> Tuple[bool, float, Dict[str, Any]]:
    """Run training for a single trial.

    Returns:
        (success, best_val_loss, result_dict)
    """
    try:
        # Ensure config_path is a Path object
        if isinstance(config_path, str):
            config_path = Path(config_path)

        # Load config to get the actual model name used by train.py
        trial_config = load_toml(config_path)
        actual_model_name = trial_config.get("model", {}).get("name", f"{model_key}_regressor")

        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train.py"),
            "--config", str(config_path)
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per trial
            cwd=str(REPO_ROOT)
        )

        if result.returncode != 0:
            print(f"Trial {trial_num} failed: {result.stderr[:500]}")
            return False, float('inf'), {}

        # Find output directory and read result
        # The output is in outputs/<actual_model_name>/<timestamp>/
        output_base = REPO_ROOT / "outputs" / actual_model_name
        if not output_base.exists():
            print(f"Trial {trial_num}: Output directory not found: {output_base}")
            return False, float('inf'), {}

        # Find most recent subdirectory
        subdirs = sorted(
            [d for d in output_base.iterdir() if d.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if not subdirs:
            print(f"Trial {trial_num}: No subdirectories found in {output_base}")
            return False, float('inf'), {}

        latest_output = subdirs[0]
        result_toml_path = latest_output / "result.toml"

        if not result_toml_path.exists():
            print(f"Trial {trial_num}: result.toml not found in {latest_output}")
            return False, float('inf'), {}

        # Parse result.toml
        result_data = load_toml(result_toml_path)
        best_val_loss = result_data.get("eval", {}).get("best_val_loss", float('inf'))

        return True, best_val_loss, {"output_dir": latest_output, "result": result_data}

    except subprocess.TimeoutExpired:
        print(f"Trial {trial_num} timed out")
        return False, float('inf'), {}
    except Exception as e:
        print(f"Trial {trial_num} exception: {e}")
        import traceback
        traceback.print_exc()
        return False, float('inf'), {}


def train_best_model(
    best_config: Dict[str, Any],
    model_name: str,
    output_dir: Path,
) -> Tuple[bool, Dict[str, Any]]:
    """Train the best model from bayesian optimization."""
    try:
        # Write best config to file
        best_config_path = output_dir / "best_config.toml"
        save_toml(best_config_path, best_config)

        # Run training
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train.py"),
            "--config", str(best_config_path)
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout for final training
            cwd=str(REPO_ROOT)
        )

        if result.returncode != 0:
            print(f"Final training failed: {result.stderr}")
            return False, {}

        # Find output directory
        output_base = REPO_ROOT / "outputs" / model_name
        if not output_base.exists():
            return False, {}

        subdirs = sorted(
            [d for d in output_base.iterdir() if d.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if not subdirs:
            return False, {}

        latest_output = subdirs[0]

        # Copy files to our output directory
        for file in latest_output.iterdir():
            if file.is_file():
                shutil.copy(file, output_dir / file.name)

        # Read result.toml
        result_toml_path = latest_output / "result.toml"
        if result_toml_path.exists():
            result_data = load_toml(result_toml_path)
            return True, result_data

        return True, {}

    except Exception as e:
        print(f"Final training exception: {e}")
        traceback.print_exc()
        return False, {}


def test_best_model(
    model_dir: Path,
    test_csv: Path,
    output_dir: Path,
) -> Tuple[bool, Dict[str, Any]]:
    """Test the best model on test data."""
    try:
        # Find the model output directory
        subdirs = sorted(
            [d for d in model_dir.iterdir() if d.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if not subdirs:
            print(f"No model directories found in {model_dir}")
            return False, {}

        latest_model_dir = subdirs[0]

        # Run test script
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "test.py"),
            "--model-dir", str(latest_model_dir),
            "--test-csv", str(test_csv),
            "--output-dir", str(output_dir),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=str(REPO_ROOT)
        )

        if result.returncode != 0:
            print(f"Test failed: {result.stderr}")
            return False, {}

        # Read predictions
        pred_csv = output_dir / "test_predictions.csv"
        if pred_csv.exists():
            df_pred = pl.read_csv(pred_csv)
            return True, {"predictions_df": df_pred}

        return True, {}

    except Exception as e:
        print(f"Test exception: {e}")
        traceback.print_exc()
        return False, {}


def run_bayesian_optimization(
    model_key: str,
    model_config: Dict[str, str],
    data_paths: Tuple[Path, Path, Path],
    output_dir: Path,
    n_trials: int,
    seed: int,
) -> Tuple[bool, Dict[str, Any]]:
    """Run bayesian optimization for a single model."""
    model_type = model_config["model_type"]
    base_config_path = REPO_ROOT / model_config["base_config"]
    bayes_config_path = REPO_ROOT / model_config["bayes_config"]

    train_csv, val_csv, test_csv = data_paths

    print(f"\n{'='*60}")
    print(f"Running Bayesian Optimization for {model_key.upper()}")
    print(f"{'='*60}")

    # Load configs
    base_config = load_toml(base_config_path)
    bayes_config = load_toml(bayes_config_path)

    # Update data paths in base config - use temporal split data
    base_config["data"]["train_csv"] = str(train_csv)
    base_config["data"]["val_csv"] = str(val_csv)
    base_config["data"]["test_csv"] = str(test_csv)

    # Create Optuna study
    study_name = f"{model_key}_temporal_optimization"
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    # Create trials directory
    trials_dir = output_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    # Store trial results
    trial_results = []

    def objective(trial: optuna.Trial) -> float:
        """Objective function for Optuna."""
        # Build config for this trial
        trial_config = deepcopy(base_config)

        # Sample parameters from bayes config
        for section_name, params in bayes_config.items():
            if section_name not in trial_config:
                trial_config[section_name] = {}

            for param_name, param_spec in params.items():
                if "values" in param_spec:
                    value = trial.suggest_categorical(
                        f"{section_name}.{param_name}",
                        param_spec["values"]
                    )
                elif "min" in param_spec and "max" in param_spec:
                    min_val = param_spec["min"]
                    max_val = param_spec["max"]
                    log_scale = param_spec.get("log", False)

                    if isinstance(min_val, int) and isinstance(max_val, int) and not log_scale:
                        value = trial.suggest_int(
                            f"{section_name}.{param_name}",
                            min_val,
                            max_val
                        )
                    else:
                        value = trial.suggest_float(
                            f"{section_name}.{param_name}",
                            min_val,
                            max_val,
                            log=log_scale
                        )
                        # Convert to int if needed
                        if isinstance(min_val, int) and isinstance(max_val, int):
                            value = int(value)
                else:
                    continue

                trial_config[section_name][param_name] = value

        # Save trial config
        trial_config_path = trials_dir / f"trial_{trial.number:03d}_config.toml"
        save_toml(trial_config_path, trial_config)

        # Save trial config to file and run training
        trial_config_path = trials_dir / f"trial_{trial.number:03d}_config.toml"
        save_toml(trial_config_path, trial_config)

        success, val_loss, result = run_training(
            trial_config_path,
            model_key,
            trial.number,
        )

        if not success or math.isnan(val_loss):
            return float('inf')

        # Store result
        trial_results.append({
            "trial_number": trial.number,
            "val_loss": val_loss,
            "config": trial_config,
        })

        return val_loss

    # Run optimization
    print(f"Running {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Check if any trials completed
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        print("No trials completed successfully.")
        return False, {}

    # Get best trial
    best_trial = study.best_trial
    best_val_loss = best_trial.value
    best_params = best_trial.params

    print(f"\nBest trial: {best_trial.number}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("Best parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # Build best config
    best_config = deepcopy(base_config)
    for param_key, param_value in best_params.items():
        section, param_name = param_key.split(".", 1)
        if section not in best_config:
            best_config[section] = {}
        best_config[section][param_name] = param_value

    # Save study results
    results_csv = output_dir / f"{model_key}_bayes_results.csv"
    with open(results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        header = ["trial_number", "value", "state"] + list(best_params.keys())
        writer.writerow(header)
        # Data
        for trial in study.trials:
            row = [trial.number, trial.value, trial.state.name]
            for param_name in best_params.keys():
                row.append(trial.params.get(param_name, ""))
            writer.writerow(row)

    print(f"\nResults saved to {output_dir}")

    return True, {
        "best_config": best_config,
        "best_val_loss": best_val_loss,
        "best_params": best_params,
        "study": study,
    }


def run_model_experiment(
    model_key: str,
    model_config: Dict[str, str],
    data_paths: Tuple[Path, Path, Path],
    output_base_dir: Path,
    n_trials: int,
    seed: int,
) -> bool:
    """Run complete experiment for a single model (bayes + final training + test)."""
    print(f"\n{'='*80}")
    print(f"Running experiment for: {model_key.upper()}")
    print(f"{'='*80}")

    model_output_dir = output_base_dir / model_key
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already completed
    completion_marker = model_output_dir / "experiment_complete.txt"
    if completion_marker.exists():
        print(f"Model {model_key} already completed. Skipping.")
        return True

    # Step 1: Bayesian Optimization
    print(f"\n[1/3] Running Bayesian Optimization ({n_trials} trials)...")
    success, bayes_result = run_bayesian_optimization(
        model_key=model_key,
        model_config=model_config,
        data_paths=data_paths,
        output_dir=model_output_dir / "bayes_opt",
        n_trials=n_trials,
        seed=seed,
    )

    if not success:
        print(f"Bayesian optimization failed for {model_key}")
        return False

    best_config = bayes_result["best_config"]
    best_val_loss = bayes_result["best_val_loss"]

    # Save best config
    best_config_path = model_output_dir / "best_config.toml"
    save_toml(best_config_path, best_config)
    print(f"Best config saved to {best_config_path}")

    # Step 2: Final Training with Best Config
    print(f"\n[2/3] Training best model...")
    final_train_dir = model_output_dir / "final_model"
    final_train_dir.mkdir(parents=True, exist_ok=True)

    train_success, train_result = train_best_model(
        best_config,
        model_key,
        final_train_dir,
    )

    if not train_success:
        print(f"Final training failed for {model_key}")
        return False

    # Step 3: Test on Test Set
    print(f"\n[3/3] Testing on test set...")
    test_output_dir = model_output_dir / "test_results"
    test_output_dir.mkdir(parents=True, exist_ok=True)

    # Get model output dir from training
    model_output_base = REPO_ROOT / "outputs" / model_key
    test_success, test_result = test_best_model(
        model_output_base,
        data_paths[2],  # test_csv
        test_output_dir,
    )

    if not test_success:
        print(f"Testing failed for {model_key}")
        # Continue anyway, we have the trained model

    # Mark as complete
    with open(completion_marker, 'w') as f:
        f.write(f"Completed at: {datetime.now().isoformat()}\n")
        f.write(f"Best val loss: {best_val_loss}\n")

    print(f"\n{model_key.upper()} experiment completed successfully!")
    print(f"Results saved to: {model_output_dir}")

    return True


def main():
    args = parse_args()

    # Set seeds
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use existing temporally-split data
    train_csv = REPO_ROOT / "data" / "train.csv"
    val_csv = REPO_ROOT / "data" / "val.csv"
    test_csv = REPO_ROOT / "data" / "test.csv"

    if not all([train_csv.exists(), val_csv.exists(), test_csv.exists()]):
        print("ERROR: Data files not found. Please run resplit_temporal_ordered.py first.")
        print(f"  Expected: {train_csv}, {val_csv}, {test_csv}")
        sys.exit(1)

    data_paths = (train_csv, val_csv, test_csv)

    print(f"{'='*80}")
    print(f"COMPREHENSIVE TEMPORAL EXPERIMENT (NO SHUFFLING)")
    print(f"{'='*80}")
    print(f"Output: {output_dir}")
    print(f"Trials per model: {args.n_trials}")
    print(f"Seed: {args.seed}")
    print(f"")
    print(f"Data (TEMPORAL ORDER maintained):")
    print(f"  Train: {train_csv} ({pd.read_csv(train_csv).shape[0]} samples)")
    print(f"  Val:   {val_csv} ({pd.read_csv(val_csv).shape[0]} samples)")
    print(f"  Test:  {test_csv} ({pd.read_csv(test_csv).shape[0]} samples)")

    # Step 1: Run experiments for each model
    print(f"\n{'='*80}")
    print(f"STEP 1: Model Experiments (Bayesian Optimization + Training + Testing)")
    print(f"{'='*80}")

    # Determine which models to run
    if "all" in args.models:
        models_to_run = list(MODEL_CONFIGS.keys())
    else:
        models_to_run = args.models

    print(f"Models to run: {', '.join(models_to_run)}")

    # Track results
    results_summary = {}
    failed_models = []

    for model_key in models_to_run:
        model_config = MODEL_CONFIGS[model_key]

        success = run_model_experiment(
            model_key=model_key,
            model_config=model_config,
            data_paths=data_paths,
            output_base_dir=output_dir / "models",
            n_trials=args.n_trials,
            seed=args.seed,
        )

        if success:
            results_summary[model_key] = "success"
        else:
            results_summary[model_key] = "failed"
            failed_models.append(model_key)

    # Step 2: Generate summary report
    print(f"\n{'='*80}")
    print(f"STEP 2: Summary Report")
    print(f"{'='*80}")

    summary = {
        "experiment_date": datetime.now().isoformat(),
        "data_split": "temporal_ordered",
        "temporal_split": True,
        "output_directory": str(output_dir),
        "n_trials": args.n_trials,
        "seed": args.seed,
        "models_run": models_to_run,
        "results": results_summary,
        "failed_models": failed_models,
    }

    summary_path = output_dir / "experiment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nExperiment Summary:")
    print(f"  Total models: {len(models_to_run)}")
    print(f"  Successful: {len(models_to_run) - len(failed_models)}")
    print(f"  Failed: {len(failed_models)}")
    if failed_models:
        print(f"  Failed models: {', '.join(failed_models)}")
    print(f"\nResults saved to: {output_dir}")
    print(f"Summary saved to: {summary_path}")

    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE TEMPORAL EXPERIMENT COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
