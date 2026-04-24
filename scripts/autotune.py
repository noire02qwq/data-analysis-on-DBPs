#!/usr/bin/env python3
"""
Unified Bayesian optimization autotune script for all supported model types.
Uses Optuna for hyperparameter tuning, reads from TOML configuration files.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import optuna
import tomli

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Bayesian optimization for model hyperparameters.")
    parser.add_argument("--model-type", required=True, help="Model type (e.g., RNN, LSTM, MLP, XGBOOST).")
    parser.add_argument("--base-config", required=True, help="Base config TOML file.")
    parser.add_argument("--bayes-config", required=True, help="Bayesian search space TOML file.")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of trials to run.")
    parser.add_argument("--output-dir", help="Output directory for results.")
    parser.add_argument("--study-name", help="Name of the Optuna study.")
    parser.add_argument("--storage", help="Storage URL for Optuna study (e.g., sqlite:///example.db).")
    return parser.parse_args()


def load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as fh:
        return tomli.load(fh)


def load_base_config(path: Path) -> Dict[str, Any]:
    return load_toml(path)


def load_bayes_config(path: Path) -> Dict[str, Any]:
    return load_toml(path)


def build_run_config(
    base_config: Dict[str, Any],
    trial: optuna.Trial,
    bayes_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a run config by sampling hyperparameters from the bayes config."""
    config = deepcopy(base_config)
    model_section = config.setdefault("model", {})
    training_section = config.setdefault("training", {})

    if not isinstance(model_section, dict) or not isinstance(training_section, dict):
        raise ValueError("Config must contain 'model' and 'training' sections.")

    # Parameters that should be integers
    INTEGER_PARAMS = {
        "history_length", "units", "num_layers", "max_depth", "min_child_samples",
        "num_leaves", "bagging_freq", "depth", "batch_size", "max_epochs",
        "nhead", "num_encoder_layers", "dim_feedforward", "d_model", "n_layers",
        "d_state", "d_conv", "expand"
    }

    # Check for "parameters" format (new style) or "model"/"training" format
    if "parameters" in bayes_config:
        # New style: [parameters.param_name] sections
        for param_name, param_spec in bayes_config["parameters"].items():
            value = _suggest_param(trial, param_name, param_spec)
            if param_name in INTEGER_PARAMS:
                value = int(value)

            # Determine which section to put the parameter in
            if param_name in ["history_length", "units", "num_layers", "dropout",
                              "max_depth", "subsample", "colsample_bytree", "gamma",
                              "reg_lambda", "reg_alpha", "min_child_weight",
                              "min_child_samples", "num_leaves", "reg_beta",
                              "boosting_type", "bagging_fraction", "bagging_freq",
                              "feature_fraction", "depth", "l2_leaf_reg",
                              "random_strength", "bagging_temperature", "nhead",
                              "num_encoder_layers", "dim_feedforward", "d_model",
                              "n_layers", "d_state", "d_conv", "expand"]:
                model_section[param_name] = value
            elif param_name in ["batch_size", "max_epochs", "learning_rate", "weight_decay", "patience", "seed"]:
                training_section[param_name] = value
            else:
                # Default to model section for unknown parameters
                model_section[param_name] = value
    else:
        # Old style: [model] and [training] sections
        # Model parameters
        model_params = bayes_config.get("model", {})
        for param_name, param_spec in model_params.items():
            value = _suggest_param(trial, param_name, param_spec)
            if param_name in INTEGER_PARAMS:
                value = int(value)
            model_section[param_name] = value

        # Training parameters
        training_params = bayes_config.get("training", {})
        for param_name, param_spec in training_params.items():
            value = _suggest_param(trial, param_name, param_spec)
            if param_name in INTEGER_PARAMS:
                value = int(value)
            training_section[param_name] = value

    return config


def _suggest_param(trial: optuna.Trial, param_name: str, param_spec: Dict[str, Any]) -> Any:
    """Suggest a parameter value based on the specification."""
    if "values" in param_spec:
        return trial.suggest_categorical(param_name, param_spec["values"])
    elif "min" in param_spec and "max" in param_spec:
        min_val = param_spec["min"]
        max_val = param_spec["max"]
        if isinstance(min_val, int) and isinstance(max_val, int):
            return trial.suggest_int(param_name, min_val, max_val)
        else:
            log = param_spec.get("log", False)
            return trial.suggest_float(param_name, min_val, max_val, log=log)
    else:
        raise ValueError(f"Invalid parameter specification for {param_name}")


def objective(
    trial: optuna.Trial,
    base_config: Dict[str, Any],
    bayes_config: Dict[str, Any],
    model_type: str,
    output_root: Path,
    run_id: str,
) -> float:
    """Objective function for Optuna optimization."""
    # Build config for this trial
    config = build_run_config(base_config, trial, bayes_config)

    # Create run directory
    run_dir = output_root / f"trial_{trial.number}_{run_id[:6]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write config file
    config_path = run_dir / "config.toml"

    # Write the full config with absolute paths so train.py can find the data
    with config_path.open("wb") as f:
        import tomli_w
        tomli_w.dump(config, f)

    # Run training
    try:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train.py"),
            "--config", str(config_path),
            "--output-dir", str(run_dir)
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=str(REPO_ROOT)
        )

        if result.returncode != 0:
            print(f"Trial {trial.number} failed: {result.stderr}")
            raise optuna.TrialPruned()

        # Training output is directly in run_dir since we passed --output-dir
        result_file = run_dir / "result.toml"

        if result_file.exists():
            result_data = load_toml(result_file)
            best_val = result_data.get("eval", {}).get("best_val_loss", float('inf'))
            if best_val is None or best_val == float('inf'):
                raise optuna.TrialPruned()
            # Save trial info including model directory for later retrieval
            trial_info_path = run_dir / "trial_info.toml"
            import tomli_w
            with trial_info_path.open("wb") as f:
                tomli_w.dump({"model_dir": str(run_dir)}, f)
            print(f"Trial {trial.number} completed with validation loss: {best_val}")
            return best_val
        else:
            print(f"Trial {trial.number}: result file not found")
            raise optuna.TrialPruned()

    except subprocess.TimeoutExpired:
        print(f"Trial {trial.number} pruned due to timeout")
        raise optuna.TrialPruned()
    except Exception as e:
        print(f"Trial {trial.number} pruned due to exception: {e}")
        error_file = run_dir / "error.txt"
        with error_file.open("w", encoding="utf-8") as f:
            f.write(f"Error: {str(e)}\n")
        raise optuna.TrialPruned()


def run_bayesian_optimization(
    model_type: str,
    base_config_path: Path,
    bayes_config_path: Path,
    n_trials: int,
    output_dir: Path | None = None,
    study_name: str | None = None,
    storage: str | None = None,
) -> None:
    """Run Bayesian optimization."""
    base_config = load_base_config(base_config_path)
    bayes_config = load_bayes_config(bayes_config_path)

    # Validate model type
    base_model_type = str(base_config.get("model", {}).get("type", "")).upper()
    if base_model_type != model_type.upper():
        raise ValueError(f"Base config model.type ({base_model_type}) does not match expected {model_type}.")

    # Set up output directory
    if output_dir is None:
        model_name = base_config.get("model", {}).get("name", "model")
        output_dir = REPO_ROOT / "outputs" / f"{model_type.lower()}_autotune"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create or load study
    if study_name is None:
        study_name = f"{model_type}_optimization"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True
    )

    run_id = uuid.uuid4().hex

    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial,
            base_config,
            bayes_config,
            model_type,
            output_dir,
            run_id
        ),
        n_trials=n_trials
    )

    # Check if any trials completed
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        print("No trials completed successfully.")
        return

    # Print best parameters
    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"Best validation loss: {study.best_value}")

    # Save study results to CSV
    results_csv = output_dir / "bayes_optimization_results.csv"
    all_param_names = set()
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            all_param_names.update(trial.params.keys())

    all_param_names = sorted(list(all_param_names))

    with results_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["trial_number", "value"] + list(all_param_names))
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                row = [trial.number, trial.value]
                for param_name in all_param_names:
                    row.append(trial.params.get(param_name, ""))
                writer.writerow(row)

    print(f"\nResults saved to {results_csv}")


def main() -> None:
    args = parse_args()

    run_bayesian_optimization(
        model_type=args.model_type,
        base_config_path=Path(args.base_config),
        bayes_config_path=Path(args.bayes_config),
        n_trials=args.n_trials,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        study_name=args.study_name,
        storage=args.storage,
    )


if __name__ == "__main__":
    main()