#!/usr/bin/env python3
"""
Fill -PPL1/-PPL2 columns by following the interpolation rules from prompt-draft/缺失填充.md.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Block:
    start: int
    end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill PPL columns using adjacent-segment regressions and noise."
    )
    parser.add_argument(
        "--input",
        default="data/raw_data.csv",
        help="CSV file with the raw readings (default: data/raw_data.csv)",
    )
    parser.add_argument(
        "--output",
        default="data/imputed_data.csv",
        help="Destination CSV for the filled readings (default: data/imputed_data.csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=947,
        help="Random seed used for the noise term (default: 42).",
    )
    return parser.parse_args()


def linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Return slope, intercept, and residual variance for a simple regression."""
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        raise ValueError("Block contains no valid samples.")

    if x.size == 1:
        slope = 0.0
        intercept = float(y[0])
        variance = 0.0
    else:
        x_mean = x.mean()
        y_mean = y.mean()
        denom = np.sum((x - x_mean) ** 2)
        slope = 0.0 if denom == 0.0 else np.sum((x - x_mean) * (y - y_mean)) / denom
        intercept = y_mean - slope * x_mean
        residuals = y - (slope * x + intercept)
        variance = float(residuals.var(ddof=1)) if residuals.size > 1 else 0.0

    return slope, intercept, max(variance, 0.0)


def extract_block(values: np.ndarray, block: Block) -> Tuple[np.ndarray, np.ndarray]:
    block_slice = slice(block.start, block.end + 1)
    x = np.arange(block.start, block.end + 1, dtype=float)
    y = values[block_slice]
    return x, y


def find_prev_block(non_nan: np.ndarray, start: int) -> Optional[Block]:
    idx = start - 1
    while idx >= 0 and not non_nan[idx]:
        idx -= 1
    if idx < 0:
        return None
    end = idx
    while idx - 1 >= 0 and non_nan[idx - 1]:
        idx -= 1
    return Block(idx, end)


def find_next_block(non_nan: np.ndarray, end: int, length: int) -> Optional[Block]:
    idx = end + 1
    while idx < length and not non_nan[idx]:
        idx += 1
    if idx >= length:
        return None
    start = idx
    while idx + 1 < length and non_nan[idx + 1]:
        idx += 1
    return Block(start, idx)


def collect_missing_segments(non_nan: np.ndarray) -> List[Block]:
    segments: List[Block] = []
    idx = 0
    length = non_nan.size
    while idx < length:
        if non_nan[idx]:
            idx += 1
            continue
        start = idx
        while idx < length and not non_nan[idx]:
            idx += 1
        segments.append(Block(start, idx - 1))
    return segments


def noise(rng: np.random.Generator, variance: float) -> float:
    if variance <= 0.0:
        return 0.0
    return float(rng.normal(0.0, math.sqrt(variance)))


def fill_segment(
    filled: np.ndarray,
    original: np.ndarray,
    segment: Block,
    prev_block: Optional[Block],
    next_block: Optional[Block],
    rng: np.random.Generator,
) -> int:
    start, end = segment.start, segment.end
    xs = np.arange(start, end + 1, dtype=float)
    filled_count = 0

    if prev_block and next_block:
        x_prev, y_prev = extract_block(original, prev_block)
        slope_prev, intercept_prev, var_prev = linear_regression(x_prev, y_prev)

        x_next, y_next = extract_block(original, next_block)
        slope_next, intercept_next, var_next = linear_regression(x_next, y_next)

        start_val = slope_prev * prev_block.end + intercept_prev
        end_val = slope_next * next_block.start + intercept_next
        denom = next_block.start - prev_block.end
        if denom <= 0:
            raise ValueError("Invalid block ordering detected.")
        noise_variance = min(var_prev, var_next)
        for idx in xs:
            weight = (idx - prev_block.end) / denom
            base = start_val + weight * (end_val - start_val)
            filled[int(idx)] = base + noise(rng, noise_variance)
            filled_count += 1
        return filled_count

    reference_block = prev_block or next_block
    if not reference_block:
        return 0

    x_ref, y_ref = extract_block(original, reference_block)
    slope, intercept, variance = linear_regression(x_ref, y_ref)
    for idx in xs:
        base = slope * idx + intercept
        filled[int(idx)] = base + noise(rng, variance)
        filled_count += 1
    return filled_count


def fill_column(values: Sequence[float], rng: np.random.Generator) -> Tuple[np.ndarray, int]:
    arr = np.asarray(values, dtype=float)
    original = arr.copy()
    filled = arr.copy()
    non_nan = ~np.isnan(original)
    segments = collect_missing_segments(non_nan)

    filled_total = 0
    for segment in segments:
        prev_block = find_prev_block(non_nan, segment.start)
        next_block = find_next_block(non_nan, segment.end, len(arr))
        filled_total += fill_segment(filled, original, segment, prev_block, next_block, rng)
    return filled, filled_total


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    target_cols = [c for c in df.columns if c.endswith("-PPL1") or c.endswith("-PPL2")]

    if not target_cols:
        raise RuntimeError("No -PPL1/-PPL2 columns found in the dataset.")

    rng = np.random.default_rng(args.seed)
    summary = []

    for col in target_cols:
        numeric = pd.to_numeric(df[col], errors="coerce")
        filled_values, count = fill_column(numeric.to_numpy(), rng)
        df[col] = filled_values
        summary.append(f"{col}: filled {count} values")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Imputed columns: {', '.join(target_cols)}")
    for line in summary:
        print(line)
    print(f"Saved filled data to {output_path}")


if __name__ == "__main__":
    main()
