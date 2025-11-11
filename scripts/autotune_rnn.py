#!/usr/bin/env python3
"""
Grid-search autotuning driver for the RNN model defined in 回归实验1109.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.autotune_runner import run_autotune  # noqa: E402

DEFAULT_BASE_CONFIG = Path("models/configs/rnn_config.yaml")
DEFAULT_GRID_CONFIG = Path("models/configs/rnn_grid.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RNN autotuning grid search.")
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG), help="Base RNN config to tweak.")
    parser.add_argument("--grid-config", default=str(DEFAULT_GRID_CONFIG), help="Grid spec YAML for tunable params.")
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Optional cap on the number of grid points to run (default: run entire grid).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip the first N grid points (useful for resuming at a specific offset).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_autotune(
        model_type="RNN",
        base_config_path=Path(args.base_config),
        grid_config_path=Path(args.grid_config),
        max_trials=args.max_trials,
        start_index=args.start_index,
    )


if __name__ == "__main__":
    main()
