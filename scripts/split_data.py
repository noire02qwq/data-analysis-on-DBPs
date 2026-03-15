#!/usr/bin/env python3
"""
Split a dataset CSV into train, validation, and test sets by row count.
Supports optional randomization (shuffling) before splitting.
Uses Polars instead of Pandas.
"""

import argparse
import random
from pathlib import Path

import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split dataset CSV into train/val/test sets.")
    parser.add_argument("--input", required=True, help="Path to the input CSV file.")
    parser.add_argument("--train-rows", type=int, required=True, help="Number of rows for the training set.")
    parser.add_argument("--val-rows", type=int, required=True, help="Number of rows for the validation set.")
    parser.add_argument("--test-rows", type=int, required=True, help="Number of rows for the test set.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the split CSVs.")
    parser.add_argument("--shuffle", action="store_true", help="Randomly shuffle rows before splitting.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling (default: 42).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {input_path}...")
    df = pl.read_csv(input_path, encoding="utf-8-sig")
    total_rows = len(df)

    requested_rows = args.train_rows + args.val_rows + args.test_rows
    if requested_rows > total_rows:
        raise ValueError(
            f"Requested {requested_rows} rows (train={args.train_rows}, val={args.val_rows}, "
            f"test={args.test_rows}), but the dataset only has {total_rows} rows."
        )

    if args.shuffle:
        print(f"Shuffling dataset with seed {args.seed}...")
        df = df.sample(n=total_rows, seed=args.seed)

    # Perform splits
    train_end = args.train_rows
    val_end = train_end + args.val_rows
    test_end = val_end + args.test_rows

    df_train = df.head(train_end)
    df_val = df.slice(train_end, args.val_rows)
    df_test = df.slice(val_end, args.test_rows)

    train_out = output_dir / "train.csv"
    val_out = output_dir / "val.csv"
    test_out = output_dir / "test.csv"

    print("Saving split datasets...")
    df_train.write_csv(train_out)
    df_val.write_csv(val_out)
    df_test.write_csv(test_out)

    print("\nDataset split complete:")
    print(f"  Training set:   {len(df_train)} rows -> {train_out}")
    print(f"  Validation set: {len(df_val)} rows -> {val_out}")
    print(f"  Test set:       {len(df_test)} rows -> {test_out}")

    if requested_rows < total_rows:
        print(f"  (Ignored remaining {total_rows - requested_rows} rows)")


if __name__ == "__main__":
    main()