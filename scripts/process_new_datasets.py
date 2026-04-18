#!/usr/bin/env python3
"""
Process new datasets (CAWW_35C, LSWW_29C, LSWW_35C):
1. Convert xlsx to csv
2. Run imputation
3. Split data 70:15:15

Usage:
    python scripts/process_new_datasets.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process new datasets: xlsx -> csv -> impute -> split"
    )
    parser.add_argument(
        "--source-dir",
        default=str(DATA_DIR),
        help="Source directory containing xlsx files files",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DATA_DIR),
        help="Output directory for processed files",
    )
    return parser.parse_args()


def xlsx_to_csv(xlsx_path: Path, csv_path: Path) -> bool:
    """Convert xlsx to csv using pandas."""
    try:
        df = pd.read_excel(xlsx_path, engine='openpyxl')
        df.to_csv(csv_path, index=False)
        print(f"  {xlsx_path.name} -> {csv_path.name}")
        return True
    except ImportError:
        print("  pandas or openpyxl not available")
        return False
    except Exception as e:
        print(f"  Error converting {xlsx_path}: {e}")
        return False


def run_imputation(input_csv: Path, output_csv: Path) -> bool:
    """Run fill_missing.py script."""
    try:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "fill_missing.py"),
            "--input", str(input_csv),
            "--output", str(output_csv),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"  Imputed: {output_csv.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Imputation failed: {e.stderr}")
        return False


def run_split(
    input_csv: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> bool:
    """Run split_data.py script."""
    try:
        # First calculate row counts
        df = pd.read_csv(input_csv)
        n_total = len(df)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = int(n_total * test_ratio)

        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "split_data.py"),
            "--input", str(input_csv),
            "--output-dir", str(output_dir),
            "--shuffle",
            "--seed", "42",
            "--train-rows", str(n_train),
            "--val-rows", str(n_val),
            "--test-rows", str(n_test),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"  Split: {n_train} train, {n_val} val, {n_test} test")
        return True
    except Exception as e:
        print(f"  Split failed: {e}")
        return False


def process_dataset(
    xlsx_path: Path,
    output_dir: Path,
    base_name: str,
) -> bool:
    """Process a single dataset through the full pipeline."""
    print(f"\nProcessing: {xlsx_path.name}")
    print(f"{'='*60}")

    # Step 1: Convert xlsx to csv
    raw_csv = output_dir / f"{base_name}_raw_data.csv"
    if not xlsx_to_csv(xlsx_path, raw_csv):
        return False

    # Step 2: Impute
    imputed_csv = output_dir / f"{base_name}_imputed_data.csv"
    if not run_imputation(raw_csv, imputed_csv):
        return False

    # Step 3: Split
    split_dir = output_dir / f"{base_name}_split"
    split_dir.mkdir(exist_ok=True)
    if not run_split(imputed_csv, split_dir):
        return False

    print(f"  Done: {base_name}")
    return True


def main():
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    datasets = [
        ("CAWW_35C_DT_full.xlsx", "caww_35c"),
        ("LSWW_29C_DT_full.xlsx", "lsww_29c"),
        ("LSWW_35C_DT_full.xlsx", "lsww_35c"),
    ]

    print("="*60)
    print("PROCESSING NEW DATASETS")
    print("="*60)

    success_count = 0
    for xlsx_name, base_name in datasets:
        xlsx_path = source_dir / xlsx_name
        if xlsx_path.exists():
            if process_dataset(xlsx_path, output_dir, base_name):
                success_count += 1
        else:
            print(f"  File not found: {xlsx_path}")

    print(f"\n{'='*60}")
    print(f"Completed: {success_count}/{len(datasets)} datasets")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
