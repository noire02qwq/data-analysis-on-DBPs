#!/usr/bin/env python3
"""
Process all new datasets (CAWW_35C, LSWW_29C, LSWW_35C):
1. Reformat xlsx to CSV matching original format
2. Run imputation (fill missing PPL columns)
3. Split data 70:15:15 (train:val:test)

Usage:
    python scripts/process_all_new_datasets.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process all new datasets: reformat -> impute -> split"
    )
    parser.add_argument(
        "--source-dir",
        default=str(DATA_DIR),
        help="Source directory containing xlsx files",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DATA_DIR),
        help="Output directory for processed files",
    )
    parser.add_argument(
        "--skip-reformat",
        action="store_true",
        help="Skip reformatting step (use existing CSV files)",
    )
    return parser.parse_args()


def fix_column_header(csv_path: Path) -> None:
    """Fix the column header to match original format exactly."""
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    # Replace the malformed header
    if '"""Date, Time"""' in content:
        content = content.replace('"""Date, Time"""', '"Date, Time"')
        with open(csv_path, 'w', encoding='utf-8-sig') as f:
            f.write(content)
        print(f"  Fixed column header in {csv_path.name}")


def reformat_excel_to_csv(xlsx_path: Path, csv_path: Path) -> bool:
    """Reformat Excel dataset to standard CSV format."""
    print(f"    Reformatting: {xlsx_path.name} -> {csv_path.name}")
    try:
        wb = openpyxl.load_workbook(xlsx_path, data_only=True)
        ws = wb.active

        # Get all rows
        all_rows = list(ws.iter_rows(values_only=True))
        print(f"      Total rows: {len(all_rows)}")

        # Expected output columns (matching original raw_data.csv)
        output_columns = [
            '"Date, Time"',
            'TRC-DT', 'TRC-RT', 'TRC-PPL1', 'TRC-PPL2',
            'pH-DT', 'pH-RT', 'pH-PPL1', 'pH-PPL2',
            'cond-DT', 'cond-RT', 'cond-PPL1', 'cond-PPL2',
            'fDOM-RT', 'fDOM-PPL1', 'fDOM-PPL2',
            'DO-RT', 'DO-PPL1', 'DO-PPL2',
            'TOC-RT', 'TOC-PPL1', 'TOC-PPL2',
            'DOC-RT', 'DOC-PPL1', 'DOC-PPL2',
        ]

        output_data = []

        # Skip first 2 rows (headers), start from row 2 (index 2)
        for i, row in enumerate(all_rows[2:], start=2):
            if row[1] is None:
                continue  # Skip rows without date

            new_row = []

            # Format date/time
            dt_val = row[1]
            if isinstance(dt_val, datetime):
                ts_str = dt_val.strftime("%Y/%m/%d %H:%M")
            else:
                ts_str = str(dt_val)
            new_row.append(ts_str)

            # TRC (cols 4-7)
            for j in [4, 5, 6, 7]:
                val = row[j] if j < len(row) else None
                new_row.append('' if val is None or (isinstance(val, str) and val.startswith('=')) else val)

            # pH (cols 8-11)
            for j in [8, 9, 10, 11]:
                val = row[j] if j < len(row) else None
                new_row.append('' if val is None or (isinstance(val, str) and val.startswith('=')) else val)

            # Conductivity (cols 12-15)
            for j in [12, 13, 14, 15]:
                val = row[j] if j < len(row) else None
                new_row.append('' if val is None or (isinstance(val, str) and val.startswith('=')) else val)

            # fDOM (cols 16-18)
            for j in [16, 17, 18]:
                val = row[j] if j < len(row) else None
                new_row.append('' if val is None or (isinstance(val, str) and val.startswith('=')) else val)

            # DO (cols 19-21)
            for j in [19, 20, 21]:
                val = row[j] if j < len(row) else None
                new_row.append('' if val is None or (isinstance(val, str) and val.startswith('=')) else val)

            # TOC (cols 22-24)
            for j in [22, 23, 24]:
                val = row[j] if j < len(row) else None
                new_row.append('' if val is None or (isinstance(val, str) and val.startswith('=')) else val)

            # DOC - use TOC values as approximation
            for j in [22, 23, 24]:
                val = row[j] if j < len(row) else None
                new_row.append('' if val is None or (isinstance(val, str) and val.startswith('=')) else val)

            output_data.append(new_row)

        # Create DataFrame
        output_df = pd.DataFrame(output_data, columns=output_columns)

        # Save to CSV
        output_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        # Fix column header if needed
        fix_column_header(csv_path)

        print(f"      Output: {len(output_df)} rows, {len(output_columns)} columns")
        return True

    except Exception as e:
        print(f"      Error: {e}")
        return False


def run_imputation(input_csv: Path, output_csv: Path) -> bool:
    """Run fill_missing.py script."""
    print(f"    Running imputation: {input_csv.name} -> {output_csv.name}")
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
            timeout=300,  # 5 minute timeout
            check=True
        )
        print(f"      Imputation successful")
        return True
    except subprocess.CalledProcessError as e:
        print(f"      Imputation failed: {e.stderr[:200]}")
        return False
    except subprocess.TimeoutExpired:
        print(f"      Imputation timed out")
        return False
    except Exception as e:
        print(f"      Imputation error: {e}")
        return False


def run_split(
    input_csv: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> bool:
    """Run split_data.py script."""
    print(f"    Splitting data: {input_csv.name}")
    try:
        # First calculate row counts
        df = pd.read_csv(input_csv, encoding='utf-8-sig')
        n_total = len(df)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = int(n_total * test_ratio)

        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "split_data.py"),
            "--input", str(input_csv),
            "--output-dir", str(output_dir),
            "--train-rows", str(n_train),
            "--val-rows", str(n_val),
            "--test-rows", str(n_test),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            check=True
        )
        print(f"      Split: {n_train} train, {n_val} val, {n_test} test")
        return True
    except Exception as e:
        print(f"      Split failed: {e}")
        return False


def process_dataset(
    xlsx_path: Path,
    output_dir: Path,
    base_name: str,
    skip_reformat: bool = False,
) -> bool:
    """Process a single dataset through the full pipeline."""
    print(f"\n  Processing: {xlsx_path.name}")
    print(f"  {'='*50}")

    # Step 1: Reformat xlsx to csv (if not skipped)
    raw_csv = output_dir / f"{base_name}_raw_data.csv"
    if not skip_reformat:
        if not reformat_excel_to_csv(xlsx_path, raw_csv):
            return False
    elif not raw_csv.exists():
        print(f"    Error: Raw CSV not found and reformatting skipped")
        return False
    else:
        print(f"    Using existing: {raw_csv.name}")

    # Step 2: Impute
    imputed_csv = output_dir / f"{base_name}_imputed_data.csv"
    if not run_imputation(raw_csv, imputed_csv):
        return False

    # Step 3: Split
    split_dir = output_dir / f"{base_name}_split"
    split_dir.mkdir(exist_ok=True)
    if not run_split(imputed_csv, split_dir):
        return False

    print(f"  ✓ Completed: {base_name}")
    return True


def main():
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    datasets = [
        ("CAWW_35C_DT_full.xlsx", "caww_35c"),
        ("LSWW_29C_DT_full.xlsx", "lsww_29c"),
        ("LSWW_35C_DT_full.xlsx", "lsww_35c"),
    ]

    print("="*60)
    print("PROCESSING ALL NEW DATASETS")
    print("="*60)

    success_count = 0
    for xlsx_name, base_name in datasets:
        xlsx_path = source_dir / xlsx_name
        if xlsx_path.exists():
            if process_dataset(
                xlsx_path,
                output_dir,
                base_name,
                skip_reformat=args.skip_reformat,
            ):
                success_count += 1
        else:
            print(f"\n  File not found: {xlsx_path}")

    print(f"\n{'='*60}")
    print(f"SUMMARY: {success_count}/{len(datasets)} datasets processed successfully")
    print(f"{'='*60}")

    if success_count == len(datasets):
        print("\nAll datasets processed successfully!")
        print(f"Output directory: {output_dir}")
        for _, base_name in datasets:
            print(f"  {base_name}:")
            print(f"    - {base_name}_raw_data.csv")
            print(f"    - {base_name}_imputed_data.csv")
            print(f"    - {base_name}_split/ (train.csv, val.csv, test.csv)")
    else:
        print("\nSome datasets failed to process. Check logs above.")


if __name__ == "__main__":
    main()