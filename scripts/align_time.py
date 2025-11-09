#!/usr/bin/env python3
"""
Align DT/RT/PPL1/PPL2 measurements so each row reflects the same batch according to
the 25/38/55-hour travel times described in prompt-draft/时间对齐.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

# Travel time in hours for a batch to move from DT to each downstream sensor.
MEASUREMENT_OFFSET_HOURS: Dict[str, int] = {
    "DT": 0,
    "RT": 25,
    "PPL1": 38,
    "PPL2": 55,
}


def offset_minutes_map() -> Dict[str, int]:
    return {suffix: hours * 60 for suffix, hours in MEASUREMENT_OFFSET_HOURS.items()}


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


def align_columns(
    df: pd.DataFrame,
    timestamp_col: str,
) -> pd.DataFrame:
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col)
    df = df.set_index(timestamp_col)

    aligned_columns: Dict[str, pd.Series] = {}
    offsets = offset_minutes_map()
    for col in df.columns:
        suffix = col.split("-")[-1]
        offset_minutes = offsets.get(suffix, 0)
        series = df[col]
        if offset_minutes:
            delta = pd.to_timedelta(offset_minutes, unit="m")
            series = series.shift(freq=-delta)
        aligned_columns[col] = series

    aligned_df = pd.DataFrame(aligned_columns, index=df.index)
    aligned_df = aligned_df.dropna().reset_index().rename(columns={"index": timestamp_col})
    return aligned_df


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
