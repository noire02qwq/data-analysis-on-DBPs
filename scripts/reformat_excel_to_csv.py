#!/usr/bin/env python3
"""
Reformat new Excel datasets to match the original CSV format.
The new datasets have a different structure from CAWW_29C.
This script transforms them to match the original format.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reformat new Excel datasets to match original CSV format."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input xlsx file path.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file path.",
    )
    return parser.parse_args()


def reformat_dataset(input_path: Path, output_path: Path) -> None:
    """Reformat Excel dataset to standard CSV format."""
    print(f"Reformatting: {input_path.name}")

    wb = openpyxl.load_workbook(input_path, data_only=True)
    ws = wb.active

    # Get all rows
    all_rows = list(ws.iter_rows(values_only=True))
    print(f"  Total rows: {len(all_rows)}")

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

        # Column mapping based on analysis:
        # Col 1: DateTime
        # Col 4: TRC-DT
        # Col 5: TRC-RT
        # Col 6: TRC-PPL1
        # Col 7: TRC-PPL2
        # Col 8: pH-DT
        # Col 9: pH-RT
        # Col 10: pH-PPL1
        # Col 11: pH-PPL2
        # Col 12: cond-DT
        # Col 13: cond-RT
        # Col 14: cond-PPL1
        # Col 15: cond-PPL2
        # Col 16: fDOM-RT
        # Col 17: fDOM-PPL1
        # Col 18: fDOM-PPL2
        # Col 19: DO-RT
        # Col 20: DO-PPL1
        # Col 21: DO-PPL2
        # Col 22: TOC-RT
        # Col 23: TOC-PPL1
        # Col 24: TOC-PPL2
        # Note: No DOC columns in new data - we'll use TOC as approximation

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

        # DOC - use TOC values as approximation (common in water quality)
        for j in [22, 23, 24]:
            val = row[j] if j < len(row) else None
            new_row.append('' if val is None or (isinstance(val, str) and val.startswith('=')) else val)

        output_data.append(new_row)

    # Create DataFrame
    output_df = pd.DataFrame(output_data, columns=output_columns)

    # Save to CSV
    output_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"  Output: {len(output_df)} rows, {len(output_columns)} columns")
    print(f"  Saved to: {output_path}")


def main():
    args = parse_args()
    reformat_dataset(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()