#!/usr/bin/env python3
"""
Bayesian optimization driver using Optuna for all supported model types.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import uuid
from copy import deepcopy
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import optuna
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

getcontext().prec = 12

GBDT_MODEL_TYPES = {"XGBOOST", "LIGHTGBM", "CATBOOST"}


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)


def build_run_config(
    base_config: Dict[str, Any],
    trial: optuna.Trial,
    grid_config: Dict[str, Any],
    model_name: str,
    target_model_type: str,
) -> Dict[str, Any]:
    config = deepcopy(base_config)
    model_section = config.setdefault("model", {})
    training_section = config.setdefault("training", {})
    if not isinstance(model_section, dict) or not isinstance(training_section, dict):
        raise ValueError("Config must contain 'model' and 'training' sections.")
    model_section["name"] = model_name

    # Define parameters that must be integers
    INTEGER_PARAMS = {
        "history_length", "units", "num_layers", "max_depth", "min_child_samples", 
        "num_leaves", "bagging_freq", "depth", "l2_leaf_reg", "batch_size", 
        "max_epochs", "nhead", "num_encoder_layers", "dim_feedforward", "mid_layer_count", 
        "mid_layer_size"
    }

    # Suggest parameters for the trial
    for param_name, param_spec in grid_config.get("parameters", {}).items():
        if "values" in param_spec:
            # Categorical parameter
            value = trial.suggest_categorical(param_name, param_spec["values"])
        elif "min" in param_spec and "max" in param_spec:
            # Numerical parameter
            min_val = param_spec["min"]
            max_val = param_spec["max"]
            
            if isinstance(min_val, int) and isinstance(max_val, int) and \
               param_spec.get("step", 1) == 1:
                # Integer parameter with step 1
                value = trial.suggest_int(param_name, min_val, max_val)
            else:
                # Float parameter or non-unit step
                if param_spec.get("log", False):
                    value = trial.suggest_float(param_name, min_val, max_val, log=True)
                else:
                    value = trial.suggest_float(param_name, min_val, max_val)
                    
                # Convert to integer if the parameter requires it
                if param_name in INTEGER_PARAMS:
                    value = int(value)
        else:
            raise ValueError(f"Invalid parameter specification for {param_name}")

        # Assign the parameter to the appropriate section
        if param_name in {"history_length", "units", "num_layers", "dropout", "max_depth", "subsample", 
                          "colsample_bytree", "gamma", "reg_lambda", "reg_alpha", "min_child_weight", 
                          "min_child_samples", "num_leaves", "reg_beta", "boosting_type", "bagging_fraction", 
                          "bagging_freq", "feature_fraction", "depth", "l2_leaf_reg", "random_strength", 
                          "bagging_temperature", "nhead", "num_encoder_layers", "dim_feedforward",
                          "mid_layer_count", "mid_layer_size"}:
            model_section[param_name] = value
        elif param_name == "learning_rate":
            if target_model_type in GBDT_MODEL_TYPES:
                model_section[param_name] = value
            else:
                training_section[param_name] = value
        elif param_name in {"batch_size", "weight_decay", "max_epochs"}:
            training_section[param_name] = value
        else:
            model_section[param_name] = value

    return config


def run_training(config_path: Path, training_entrypoint: str) -> None:
    cmd = ["python", training_entrypoint, "--config", str(config_path)]
    result = subprocess.run(cmd, check=False, cwd=REPO_ROOT)  # 不再使用check=True
    if result.returncode != 0:
        print(f"Training script failed with return code {result.returncode}")
        raise subprocess.CalledProcessError(result.returncode, cmd)


def read_best_val_loss(metadata_path: Path) -> float:
    try:
        with metadata_path.open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)
        history = metadata.get("training_history", {})
        if isinstance(history, dict):
            # 优先使用primary_best_val_loss（如果有），否则使用best_val_loss
            primary = history.get("primary_best_val_loss")
            if isinstance(primary, (int, float)):
                return float(primary)
            best_val = history.get("best_val_loss")
            if isinstance(best_val, (int, float)):
                return float(best_val)
        print(f"Invalid training history in {metadata_path}: {history}")
        return math.nan
    except FileNotFoundError:
        print(f"Metadata file not found: {metadata_path}")
        return math.nan
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {metadata_path}: {e}")
        return math.nan
    except Exception as e:
        # 如果读取或解析失败，记录错误并返回NaN
        print(f"Error reading best validation loss from {metadata_path}: {e}")
        return math.nan


def objective(
    trial: optuna.Trial,
    base_config: Dict[str, Any],
    grid_config: Dict[str, Any],
    model_type: str,
    output_root: Path,
    training_script: str,
    model_name_prefix: str = "",
) -> float:
    base_model = base_config.get("model")
    if not isinstance(base_model, dict):
        raise ValueError("Base config missing 'model' section.")
    base_model_name = str(base_model.get("name") or "").split("/")[0].strip()
    if not base_model_name:
        raise ValueError("Base config must define model.name.")

    effective_model_name = base_model_name
    if model_name_prefix and not effective_model_name.startswith(model_name_prefix):
        effective_model_name = f"{model_name_prefix}{effective_model_name}"

    # Create a temporary directory for this trial
    temp_label = f"trial_{trial.number}_{uuid.uuid4().hex[:6]}"
    run_dir = output_root / temp_label
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Build config for this trial
        config_path = run_dir / "config.yaml"
        # 修复model_name生成逻辑，确保与model_name_prefix一致
        if model_name_prefix:
            # 直接使用model_name_prefix作为基础名称，避免重复后缀
            model_name = f"{model_name_prefix}/{temp_label}"
        else:
            model_name = f"{base_model_name}/{temp_label}"
            
        run_config = build_run_config(base_config, trial, grid_config, model_name, model_type)
        
        # Save config
        save_yaml(config_path, run_config)
        
        # Run training
        run_training(config_path, training_script)
        
        # Read best validation loss
        metadata_path = run_dir / "metadata.json"
        best_val = read_best_val_loss(metadata_path)
        
        # If validation loss is NaN, treat as a failed trial
        if math.isnan(best_val):
            print(f"Trial {trial.number} pruned due to NaN validation loss")
            raise optuna.TrialPruned()
            
        print(f"Trial {trial.number} completed with validation loss: {best_val}")
            
        return best_val
    except subprocess.CalledProcessError:
        print(f"Trial {trial.number} pruned due to subprocess error")
        raise optuna.TrialPruned()
    except FileNotFoundError:
        print(f"Trial {trial.number} pruned due to file not found")
        raise optuna.TrialPruned()
    # 注意：我们不再在finally块中清理run_dir，这样可以保留每个trial的结果
    # 但我们需要确保在出现异常时记录相关信息
    except Exception as e:
        # 记录异常信息到文件
        print(f"Trial {trial.number} pruned due to exception: {e}")
        error_file = run_dir / "error.txt"
        with error_file.open("w", encoding="utf-8") as f:
            f.write(f"Error: {str(e)}\n")
        raise optuna.TrialPruned()


def run_bayesian_optimization(
    *,
    model_type: str,
    base_config_path: Path,
    grid_config_path: Path,
    n_trials: int,
    training_script: str = "scripts/train_regression.py",
    model_name_prefix: str = "",
    study_name: str | None = None,
    storage: str | None = None,
) -> None:
    base_config = load_yaml(base_config_path)
    grid_config = load_yaml(grid_config_path)
    
    base_model = base_config.get("model")
    if not isinstance(base_model, dict):
        raise ValueError("Base config missing 'model' section.")
    base_model_name = str(base_model.get("name") or "").split("/")[0].strip()
    if not base_model_name:
        raise ValueError("Base config must define model.name.")
    base_model_type = str(base_model.get("type") or "").upper()
    if base_model_type != model_type.upper():
        raise ValueError(f"Base config model.type ({base_model_type}) does not match expected {model_type}.")

    effective_model_name = base_model_name
    if model_name_prefix and not effective_model_name.startswith(model_name_prefix):
        effective_model_name = f"{model_name_prefix}{effective_model_name}"

    # 根据命名规范修改输出目录：模型-输出目标-回归目标
    # 从model_name_prefix中提取模型类型、输出目标和回归目标
    if model_name_prefix:
        # 移除末尾的连字符
        prefix_parts = model_name_prefix.rstrip('-').split('-')
        if len(prefix_parts) >= 3:
            model_type_name = prefix_parts[0]
            output_target = prefix_parts[1]
            regression_target = prefix_parts[2]
            # 统一输出目录命名，所有trial都放在统一的目录下
            effective_output_name = f"{model_type_name}-{output_target}-{regression_target}"
        else:
            effective_output_name = effective_model_name
    else:
        effective_output_name = effective_model_name
    
    output_root = REPO_ROOT / "scripts" / "outputs" / effective_output_name
    output_root.mkdir(parents=True, exist_ok=True)

    # Create or load study
    if study_name is None:
        study_name = f"{model_type}_optimization"
        
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial,
            base_config,
            grid_config,
            model_type,
            output_root,
            training_script,
            model_name_prefix
        ),
        n_trials=n_trials
    )

    # Check if any trials completed successfully
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        print("No trials completed successfully.")
        return

    # Print best parameters
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"Best validation loss: {study.best_value}")

    # Save study results to CSV
    results_csv = output_root / "bayes_optimization_results.csv"
    # 收集所有试验中出现过的参数名称
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

    print(f"Results saved to {results_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Bayesian optimization for model hyperparameters.")
    parser.add_argument("--model-type", required=True, help="Model type (e.g., LSTM, MLP, XGBOOST).")
    parser.add_argument("--base-config", required=True, help="Base config YAML file.")
    parser.add_argument("--grid-config", required=True, help="Grid config YAML file with parameter ranges.")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of trials to run.")
    parser.add_argument("--training-script", default="scripts/train_regression.py", help="Training script to use.")
    parser.add_argument("--model-name-prefix", default="", help="Prefix for model name.")
    parser.add_argument("--study-name", help="Name of the Optuna study.")
    parser.add_argument("--storage", help="Storage URL for Optuna study (e.g., sqlite:///example.db).")
    
    args = parser.parse_args()
    
    run_bayesian_optimization(
        model_type=args.model_type,
        base_config_path=Path(args.base_config),
        grid_config_path=Path(args.grid_config),
        n_trials=args.n_trials,
        training_script=args.training_script,
        model_name_prefix=args.model_name_prefix,
        study_name=args.study_name,
        storage=args.storage,
    )


if __name__ == "__main__":
    main()