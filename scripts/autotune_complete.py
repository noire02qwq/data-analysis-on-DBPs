#!/usr/bin/env python3
"""
Complete Bayesian optimization with full trial outputs saved.
Each trial saves: config.toml, result.toml, loss_history.csv, best_model.pt, training_curve.png
After completion, keeps only the best trial.
"""

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

optuna.logging.set_verbosity(optuna.logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run complete Bayesian optimization with full trial outputs.")
    parser.add_argument("--model-type", required=True, help="Model type (e.g., RNN, LSTM, MLP, XGBOOST).")
    parser.add_argument("--base-config", required=True, help="Base config TOML file.")
    parser.add_argument("--bayes-config", required=True, help="Bayesian search space TOML file.")
    parser.add_argument("--n-trials", type=int, default=200, help="Number of trials to run.")
    parser.add_argument("--output-dir", help="Output directory for results.")
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

    INTEGER_PARAMS = {
        "history_length", "units", "num_layers", "max_depth", "min_child_samples",
        "num_leaves", "bagging_freq", "depth", "batch_size", "max_epochs",
        "nhead", "num_encoder_layers", "dim_feedforward", "d_model", "n_layers",
        "d_state", "d_conv", "expand", "mid_layer_count", "mid_layer_size"
    }

    if "parameters" in bayes_config:
        for param_name, param_spec in bayes_config["parameters"].items():
            value = _suggest_param(trial, param_name, param_spec)
            if param_name in INTEGER_PARAMS:
                value = int(value)

            # Special handling for MLP hidden_layers
            if param_name == "mid_layer_count" or param_name == "mid_layer_size":
                # Need both parameters to build hidden_layers
                continue

            model_section[param_name] = value

        # Build hidden_layers for MLP if mid_layer_count and mid_layer_size are present
        if "mid_layer_count" in bayes_config["parameters"] and "mid_layer_size" in bayes_config["parameters"]:
            mid_layer_count = int(_suggest_param(trial, "mid_layer_count", bayes_config["parameters"]["mid_layer_count"]))
            mid_layer_size = int(_suggest_param(trial, "mid_layer_size", bayes_config["parameters"]["mid_layer_size"]))
            # Create hidden layers list with decreasing size
            hidden_layers = [mid_layer_size // (2**i) for i in range(mid_layer_count)]
            model_section["hidden_layers"] = hidden_layers

    else:
        model_params = bayes_config.get("model", {})
        for param_name, param_spec in model_params.items():
            value = _suggest_param(trial, param_name, param_spec)
            if param_name in INTEGER_PARAMS:
                value = int(value)
            model_section[param_name] = value

        training_params = bayes_config.get("training", {})
        for param_name, param_spec in training_params.items():
            value = _suggest_param(trial, param_name, param_spec)
            if param_name in INTEGER_PARAMS:
                value = int(value)
            training_section[param_name] = value

    return config


def _suggest_param(trial: optuna.Trial, param_name: str, param_spec: Dict[str, Any]) -> Any:
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
    config = build_run_config(base_config, trial, bayes_config)
    run_dir = output_root / f"trial_{trial.number}_{run_id[:6]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.toml"
    config_for_file = deepcopy(config)
    if "data" in config_for_file:
        for key in ["train_csv", "val_csv", "test_csv"]:
            if key in config_for_file["data"]:
                config_for_file["data"][key] = f"data/{Path(config_for_file['data'][key]).name}"

    with config_path.open("wb") as f:
        import tomli_w
        tomli_w.dump(config_for_file, f)

    try:
        model_name = config.get("model", {}).get("name", f"{model_type}_trial_{trial.number}")
        output_subdir = REPO_ROOT / "outputs" / model_name

        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train.py"),
            "--config", str(config_path)
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(REPO_ROOT)
        )

        if result.returncode != 0:
            print(f"Trial {trial.number} failed: {result.stderr[-500:]}")
            raise optuna.TrialPruned()

        # Copy all outputs from the training run to the trial directory
        if output_subdir.exists():
            # Find the latest output directory
            subdirs = sorted(output_subdir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if subdirs:
                train_output = subdirs[0]

                # Copy all relevant files
                for fname in ["config.toml", "result.toml", "loss_history.csv",
                              "best_model.pt", "last_model.pt", "scalers.npz", "training_curve.png"]:
                    src = train_output / fname
                    if src.exists():
                        shutil.copy(src, run_dir / fname)

                # Also copy test_results if exists
                test_results_src = train_output / "test_results"
                if test_results_src.exists():
                    test_results_dst = run_dir / "test_results"
                    shutil.copytree(test_results_src, test_results_dst)

        result_file = run_dir / "result.toml"
        if result_file.exists():
            result_data = load_toml(result_file)
            best_val = result_data.get("eval", {}).get("best_val_loss", float('inf'))
            if best_val is None or best_val == float('inf'):
                raise optuna.TrialPruned()
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


def keep_best_trial(output_dir: Path) -> None:
    """After optimization, keep only the best trial and delete others."""
    trials = list(output_dir.glob("trial_*"))
    if not trials:
        return

    best_trial = None
    best_val_loss = float('inf')

    for trial_dir in trials:
        result_file = trial_dir / "result.toml"
        if result_file.exists():
            try:
                result_data = load_toml(result_file)
                val_loss = result_data.get("eval", {}).get("best_val_loss", float('inf'))
                if val_loss is not None and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_trial = trial_dir
            except:
                pass

    if best_trial:
        print(f"\nBest trial: {best_trial.name} with val_loss={best_val_loss:.6f}")
        print("Deleting non-best trials...")

        for trial_dir in trials:
            if trial_dir != best_trial:
                shutil.rmtree(trial_dir)
        print(f"Kept: {best_trial.name}")
    else:
        print("No valid trial found to keep")


def run_bayesian_optimization(
    model_type: str,
    base_config_path: Path,
    bayes_config_path: Path,
    n_trials: int,
    output_dir: Path | None = None,
) -> None:
    """Run Bayesian optimization with complete trial outputs."""
    base_config = load_base_config(base_config_path)
    bayes_config = load_bayes_config(bayes_config_path)

    base_model_type = str(base_config.get("model", {}).get("type", "")).upper()
    if base_model_type != model_type.upper():
        raise ValueError(f"Base config model.type ({base_model_type}) does not match expected {model_type}.")

    if output_dir is None:
        model_name = base_config.get("model", {}).get("name", "model")
        output_dir = REPO_ROOT / "outputs" / f"{model_type.lower()}_bayes_v3"
    output_dir.mkdir(parents=True, exist_ok=True)

    study_name = f"{model_type}_optimization_v3"
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        load_if_exists=True
    )

    run_id = uuid.uuid4().hex

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

    # Save results to CSV
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

    # Keep only best trial
    keep_best_trial(output_dir)

    print(f"\nResults saved to {results_csv}")
    try:
        if study.best_value:
            print(f"Best validation loss: {study.best_value:.6f}")
    except ValueError:
        print("No valid trials completed")


def main() -> None:
    args = parse_args()

    run_bayesian_optimization(
        model_type=args.model_type,
        base_config_path=Path(args.base_config),
        bayes_config_path=Path(args.bayes_config),
        n_trials=args.n_trials,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()