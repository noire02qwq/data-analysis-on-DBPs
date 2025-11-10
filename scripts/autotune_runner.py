#!/usr/bin/env python3
"""
Shared helpers for LSTM/RNN autotuning workflows.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import subprocess
from copy import deepcopy
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

import yaml

getcontext().prec = 12

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def float_range(min_value: float, max_value: float, step: float) -> List[float]:
    values: List[float] = []
    current = Decimal(str(min_value))
    max_decimal = Decimal(str(max_value))
    step_decimal = Decimal(str(step))
    epsilon = Decimal("1e-9")
    while current <= max_decimal + epsilon:
        values.append(float(current))
        current += step_decimal
    return values


def expand_parameter_values(spec: Dict[str, object]) -> List[float | int]:
    if "values" in spec:
        vals = spec["values"]
        if not isinstance(vals, Iterable):
            raise ValueError("values must be an iterable list.")
        return [v for v in vals]
    min_value = spec.get("min")
    max_value = spec.get("max")
    step = spec.get("step")
    if min_value is None or max_value is None or step is None:
        raise ValueError("Range parameters require min, max, and step.")
    if isinstance(min_value, int) and isinstance(max_value, int) and isinstance(step, int):
        return list(range(int(min_value), int(max_value) + 1, int(step)))
    return float_range(float(min_value), float(max_value), float(step))


def iter_parameter_grid(parameters: Dict[str, Dict[str, object]]) -> Iterator[Dict[str, float | int]]:
    keys = sorted(parameters.keys())
    values = [expand_parameter_values(parameters[key]) for key in keys]
    for combo in itertools.product(*values):
        yield {key: value for key, value in zip(keys, combo)}


def load_existing_run_ids(output_dir: Path) -> List[int]:
    ids: List[int] = []
    if not output_dir.exists():
        return ids
    for child in output_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            ids.append(int(child.name))
    return sorted(ids)


def load_completed_runs(results_csv: Path) -> Dict[str, Dict[str, str]]:
    completed: Dict[str, Dict[str, str]] = {}
    if not results_csv.exists():
        return completed
    with results_csv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            completed[row["run_id"]] = row
    return completed


def write_results_row(
    csv_path: Path,
    fieldnames: Sequence[str],
    row: Dict[str, object],
) -> None:
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def build_run_config(
    base_config: Dict[str, object],
    overrides: Dict[str, float | int],
    model_name: str,
) -> Dict[str, object]:
    config = deepcopy(base_config)
    model_section = config.setdefault("model", {})
    training_section = config.setdefault("training", {})
    if not isinstance(model_section, dict) or not isinstance(training_section, dict):
        raise ValueError("Config must contain 'model' and 'training' sections.")
    model_section["name"] = model_name
    for key, value in overrides.items():
        if key in {"history_length", "units", "num_layers", "dropout"}:
            model_section[key] = value
        elif key in {"batch_size", "learning_rate", "weight_decay"}:
            training_section[key] = value
        else:
            raise ValueError(f"Unknown parameter override: {key}")
    return config


def run_training(config_path: Path) -> None:
    cmd = ["python", "scripts/train_regression.py", "--config", str(config_path)]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def read_best_val_loss(metadata_path: Path) -> float:
    with metadata_path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    history = metadata.get("training_history", {})
    if isinstance(history, dict):
        best_val = history.get("best_val_loss")
        if isinstance(best_val, (int, float)):
            return float(best_val)
    return math.nan


def run_autotune(
    *,
    model_type: str,
    base_config_path: Path,
    grid_config_path: Path,
    max_trials: int | None,
    start_index: int,
) -> None:
    base_config = load_yaml(base_config_path)
    grid_config = load_yaml(grid_config_path)
    parameters = grid_config.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("Grid config must contain a 'parameters' mapping.")
    base_model = base_config.get("model")
    if not isinstance(base_model, dict):
        raise ValueError("Base config missing 'model' section.")
    base_model_name = str(base_model.get("name") or "").split("/")[0].strip()
    if not base_model_name:
        raise ValueError("Base config must define model.name.")
    base_model_type = str(base_model.get("type") or "").upper()
    if base_model_type != model_type.upper():
        raise ValueError(f"Base config model.type ({base_model_type}) does not match expected {model_type}.")

    output_root = REPO_ROOT / "scripts" / "outputs" / base_model_name
    output_root.mkdir(parents=True, exist_ok=True)
    results_csv = output_root / "autotune_results.csv"
    existing_runs = load_existing_run_ids(output_root)
    completed_rows = load_completed_runs(results_csv)
    next_run_number = existing_runs[-1] + 1 if existing_runs else 1

    fieldnames = [
        "run_id",
        "status",
        "best_val_loss",
        "history_length",
        "units",
        "num_layers",
        "dropout",
        "batch_size",
        "learning_rate",
        "weight_decay",
    ]

    trial_index = 0
    executed_trials = 0
    for combo in iter_parameter_grid(parameters):
        if trial_index < start_index:
            trial_index += 1
            continue
        run_id = f"{next_run_number:04d}"
        next_run_number += 1
        model_name = f"{base_model_name}/{run_id}"
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.yaml"
        run_config = build_run_config(base_config, combo, model_name)
        with config_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(run_config, fh, allow_unicode=True, sort_keys=False)

        status = "success"
        best_val = math.nan
        try:
            run_training(config_path)
            metadata_path = run_dir / "metadata.json"
            best_val = read_best_val_loss(metadata_path)
        except subprocess.CalledProcessError:
            status = "failed"
        except FileNotFoundError:
            status = "missing_metadata"

        row = {
            "run_id": run_id,
            "status": status,
            "best_val_loss": f"{best_val:.6f}" if isinstance(best_val, float) and not math.isnan(best_val) else "",
        }
        row.update({key: combo.get(key, "") for key in fieldnames if key in combo})
        write_results_row(results_csv, fieldnames, row)

        executed_trials += 1
        trial_index += 1
        if max_trials is not None and executed_trials >= max_trials:
            break
