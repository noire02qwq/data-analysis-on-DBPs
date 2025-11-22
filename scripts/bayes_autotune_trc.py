#!/usr/bin/env python3
"""
Bayesian optimization autotuning driver for TRC value regression targets.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bayes_optimize import run_bayesian_optimization  # noqa: E402

DEFAULT_BASE_CONFIG = Path("models/configs/lstm_config.yaml")
DEFAULT_GRID_CONFIG = Path("models/configs/lstm_bayes.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Bayesian optimization for TRC value targets.")
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG), help="Base config to tweak.")
    parser.add_argument("--grid-config", default=str(DEFAULT_GRID_CONFIG), help="Bayes spec YAML for tunable params.")
    parser.add_argument("--model-type", required=True, help="Model type (e.g., LSTM, MLP, XGBOOST).")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of trials to run (default: 100).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_bayesian_optimization(
        model_type=args.model_type,
        base_config_path=Path(args.base_config),
        grid_config_path=Path(args.grid_config),
        n_trials=args.n_trials,
        training_script="scripts/train_trc_regression.py",
        model_name_prefix=f"{args.model_type.lower()}-trc-value",
    )


if __name__ == "__main__":
    main()