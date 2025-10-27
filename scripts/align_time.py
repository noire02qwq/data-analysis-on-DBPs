#!/usr/bin/env python3
"""
Align DT/RT/PPL1/PPL2 measurements so each row represents the same fluid batch.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

OFFSET_MINUTES: Dict[str, int] = {
    "DT": 0,
    "RT": 25,
    "PPL1": 40,
    "PPL2": 55,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Drop the empty cond-RT column and realign RT/PPL1/PPL2 readings "
            "so each row captures the same batch across measurement points."
        )
    )
    parser.add_argument(
        "--input",
        default="data/imputed_data.csv",
        help="CSV produced after missing-value filling (default: data/imputed_data.csv)",
    )
    parser.add_argument(
        "--output",
        default="data/time_aligned_data.csv",
        help="Destination CSV for the aligned table (default: data/time_aligned_data.csv)",
    )
    parser.add_argument(
        "--timestamp-column",
        default="Date, Time",
        help='Name of the timestamp column that represents DT time (default: "Date, Time")',
    )
    return parser.parse_args()


def estimate_step_minutes(timestamps: pd.Series) -> float:
    diffs = timestamps.diff().dropna()
    if diffs.empty:
        raise ValueError("Unable to estimate sampling interval from a single timestamp.")
    median = diffs.median()
    minutes = median.total_seconds() / 60.0
    if minutes <= 0:
        raise ValueError("Non-positive sampling interval detected.")
    return minutes


def rows_for_offset(offset_minutes: int, step_minutes: float) -> int:
    if offset_minutes == 0:
        return 0
    periods = int(round(offset_minutes / step_minutes))
    if not np.isclose(periods * step_minutes, offset_minutes, atol=1e-6):
        raise ValueError(
            f"Offset {offset_minutes} minutes is not a multiple of the sampling interval {step_minutes} minutes."
        )
    return periods


def align_columns(
    df: pd.DataFrame,
    timestamp_col: str,
) -> pd.DataFrame:
    timestamps = pd.to_datetime(df[timestamp_col])
    df = df.copy()
    df[timestamp_col] = timestamps
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    step_minutes = estimate_step_minutes(df[timestamp_col])
    suffix_shift: Dict[str, int] = {
        suffix: rows_for_offset(offset, step_minutes) for suffix, offset in OFFSET_MINUTES.items()
    }

    data_cols = [c for c in df.columns if c != timestamp_col]
    for col in data_cols:
        suffix = col.split("-")[-1]
        shift_periods = suffix_shift.get(suffix)
        if shift_periods is None or shift_periods == 0:
            continue
        df[col] = df[col].shift(-shift_periods)

    aligned = df.dropna(subset=data_cols).reset_index(drop=True)
    return aligned


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    timestamp_col = args.timestamp_column
    if timestamp_col not in df.columns:
        raise KeyError(f"Timestamp column '{timestamp_col}' not found in the dataset.")

    cond_rt_dropped = False
    if "cond-RT" in df.columns:
        df = df.drop(columns=["cond-RT"])
        cond_rt_dropped = True

    before_rows = len(df)
    aligned_df = align_columns(df, timestamp_col=timestamp_col)
    after_rows = len(aligned_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    if cond_rt_dropped:
        print("Dropped column: cond-RT")
    print(f"Aligned data rows: {after_rows} (from {before_rows}, removed {before_rows - after_rows})")
    print(f"Saved time-aligned table to {output_path}")


if __name__ == "__main__":
    main()
