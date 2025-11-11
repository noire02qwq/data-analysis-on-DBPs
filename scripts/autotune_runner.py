#!/usr/bin/env python3
"""
Shared helpers for LSTM/RNN autotuning workflows.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import random
import shutil
import subprocess
import uuid
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


def iter_parameter_grid(
    keys: Sequence[str],
    values: Sequence[Sequence[float | int]],
) -> Iterator[Dict[str, float | int]]:
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

    keys = sorted(parameters.keys())
    value_lists: List[List[float | int]] = [expand_parameter_values(parameters[key]) for key in keys]
    value_lookup = {key: value_lists[idx] for idx, key in enumerate(keys)}
    start_values = {}
    for key, spec in parameters.items():
        start_value = spec.get("start")
        if start_value is None:
            raise ValueError(f"Parameter '{key}' must define a 'start' value.")
        if start_value not in value_lookup[key]:
            raise ValueError(f"Start value {start_value} not in grid for parameter '{key}'.")
        start_values[key] = start_value

    total_combinations = math.prod(len(values) for values in value_lists) if value_lists else 0
    print(f"[{model_type}] Total hyperparameter combinations: {total_combinations}")
    if start_index > 0:
        print(f"[{model_type}] start_index={start_index} is ignored for hill climbing; using defined start values.")

    output_root = REPO_ROOT / "scripts" / "outputs" / base_model_name
    output_root.mkdir(parents=True, exist_ok=True)
    results_csv = output_root / "autotune_results.csv"
    existing_runs = load_existing_run_ids(output_root)
    next_run_number = (existing_runs[-1] + 1) if existing_runs else 1

    param_fieldnames = [
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

    fieldnames = param_fieldnames + ["moved_param", "new_value"]

    evaluations = 0

    def should_stop() -> bool:
        return max_trials is not None and evaluations >= max_trials

    def run_state(state: Dict[str, float | int], label: str) -> tuple[float, str]:
        nonlocal evaluations
        if should_stop():
            return math.inf, "skipped"
        run_dir = output_root / label
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.yaml"
        model_name = f"{base_model_name}/{label}"
        run_config = build_run_config(base_config, state, model_name)
        with config_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(run_config, fh, allow_unicode=True, sort_keys=False)
        evaluations += 1
        try:
            run_training(config_path)
            metadata_path = run_dir / "metadata.json"
            best_val = read_best_val_loss(metadata_path)
            return best_val, "success"
        except subprocess.CalledProcessError:
            shutil.rmtree(run_dir, ignore_errors=True)
            return math.inf, "failed"
        except FileNotFoundError:
            shutil.rmtree(run_dir, ignore_errors=True)
            return math.inf, "missing_metadata"

    def record_run(
        run_id: str,
        state: Dict[str, float | int],
        status: str,
        best_val: float,
        moved_param: str | None = None,
    ) -> None:
        row = {
            "run_id": run_id,
            "status": status,
            "best_val_loss": f"{best_val:.6f}" if isinstance(best_val, float) and not math.isnan(best_val) else "",
            "moved_param": moved_param or "",
            "new_value": state.get(moved_param, "") if moved_param else "",
        }
        row.update({key: state.get(key, "") for key in param_fieldnames if key in state})
        write_results_row(results_csv, fieldnames, row)

    def finalize_dir(temp_label: str, final_run_id: str) -> None:
        temp_dir = output_root / temp_label
        final_dir = output_root / final_run_id
        if final_dir.exists():
            shutil.rmtree(final_dir)
        temp_dir.rename(final_dir)
        metadata_path = final_dir / "metadata.json"
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as fh:
                metadata = json.load(fh)
            metadata["model_name"] = f"{base_model_name}/{final_run_id}"
            model_files = metadata.get("model_files", {})
            temp_prefix = Path("scripts") / "outputs" / base_model_name / temp_label
            final_prefix = Path("scripts") / "outputs" / base_model_name / final_run_id
            for key, path_str in list(model_files.items()):
                if not path_str:
                    continue
                try:
                    relative = Path(path_str).relative_to(temp_prefix)
                    model_files[key] = str(final_prefix / relative)
                except ValueError:
                    pass
            config_path_str = metadata.get("config_path")
            if config_path_str:
                try:
                    relative = Path(config_path_str).relative_to(temp_prefix)
                    metadata["config_path"] = str(final_prefix / relative)
                except ValueError:
                    pass
            with metadata_path.open("w", encoding="utf-8") as fh:
                json.dump(metadata, fh, ensure_ascii=False, indent=2)

    current_state = dict(start_values)
    initial_run_id = f"{next_run_number:04d}"
    next_run_number += 1
    print(f"[{model_type}] Evaluating initial state as run {initial_run_id}: {current_state}")
    best_val_loss, status = run_state(current_state, initial_run_id)
    if status != "success":
        record_run(initial_run_id, current_state, status, best_val_loss)
        print(f"[{model_type}] Initial run failed; aborting autotune.")
        return
    record_run(initial_run_id, current_state, status, best_val_loss)
    print(f"[{model_type}] Initial state best_val={best_val_loss:.6f}")

    while not should_stop():
        improved = False
        param_order = keys[:]
        random.shuffle(param_order)

        for param in param_order:
            values = value_lookup[param]
            idx = values.index(current_state[param])
            neighbor_indices = []
            if idx > 0:
                neighbor_indices.append(idx - 1)
            if idx < len(values) - 1:
                neighbor_indices.append(idx + 1)
            if not neighbor_indices:
                continue

            neighbor_runs = []
            for neighbor_idx in neighbor_indices:
                candidate_state = dict(current_state)
                candidate_state[param] = values[neighbor_idx]
                temp_label = f"tmp_{param}_{uuid.uuid4().hex[:6]}"
                best_val, status = run_state(candidate_state, temp_label)
                neighbor_runs.append((best_val, status, temp_label, candidate_state))
                print(
                    f"[{model_type}]   Candidate {param} -> {candidate_state[param]} "
                    f"(val={best_val if math.isfinite(best_val) else 'nan'}) status={status}"
                )

            valid_candidates = [
                entry for entry in neighbor_runs if entry[1] == "success" and entry[0] < best_val_loss - 1e-9
            ]
            if valid_candidates:
                valid_candidates.sort(key=lambda item: item[0])
                best_candidate = valid_candidates[0]
                accepted_label = best_candidate[2]
                current_state = best_candidate[3]
                best_val_loss = best_candidate[0]

                final_run_id = f"{next_run_number:04d}"
                next_run_number += 1
                finalize_dir(accepted_label, final_run_id)
                record_run(final_run_id, current_state, "success", best_val_loss, moved_param=param)
                print(
                    f"[{model_type}] Improvement: {param} -> {current_state[param]} "
                    f"(run {final_run_id}, best_val={best_val_loss:.6f})"
                )

                for _, _, label, _ in neighbor_runs:
                    candidate_dir = output_root / label
                    if candidate_dir.exists() and label != accepted_label:
                        shutil.rmtree(candidate_dir, ignore_errors=True)

                improved = True
                break
            else:
                for _, _, label, _ in neighbor_runs:
                    candidate_dir = output_root / label
                    if candidate_dir.exists():
                        shutil.rmtree(candidate_dir, ignore_errors=True)

            if should_stop():
                break

        if not improved or should_stop():
            print(f"[{model_type}] No further improvements; stopping autotune.")
            break
