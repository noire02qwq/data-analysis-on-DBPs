#!/usr/bin/env python3
"""
Complete fix for LSWW29 data - handle nulls and add time annotation
"""
import pandas as pd
import numpy as np
import os

def impute_column(series):
    """Impute null values using forward fill then backward fill"""
    return series.ffill().bfill()

def fix_file(input_file, output_file, exclude_cols=None):
    """Fix a single file: add time annotation and impute nulls"""
    print(f"Processing: {input_file}")

    # Read CSV
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    original_shape = df.shape
    print(f"  Original shape: {original_shape}")

    # Exclude columns if specified
    if exclude_cols:
        df = df.drop(columns=[c for c in exclude_cols if c in df.columns])
        print(f"  After excluding {exclude_cols}: {df.shape}")

    # Impute nulls
    null_before = df.isnull().sum().sum()
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = impute_column(df[col])
    null_after = df.isnull().sum().sum()
    print(f"  Nulls: {null_before} -> {null_after}")

    # Add minutes_since_start
    df['minutes_since_start'] = [i * 5 for i in range(len(df))]

    # Save
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"  Final shape: {df.shape}")
    print(f"  Saved to: {output_file}")
    print()

    return df

def main():
    """Fix LSWW29 and LSWW35 data"""

    print("="*60)
    print("Fixing LSWW29 and LSWW35 Data")
    print("="*60)
    print()

    # LSWW29 - exclude DO columns (100% null)
    print("=== LSWW29 ===")
    for split in ['train', 'val', 'test']:
        input_file = f'data/lsww_29c_split/{split}_fixed.csv'
        output_file = f'data/lsww_29c_split/{split}_clean.csv'

        if os.path.exists(input_file):
            fix_file(input_file, output_file, exclude_cols=['DO-RT', 'DO-PPL1', 'DO-PPL2'])
        else:
            print(f"Warning: {input_file} not found")

    # LSWW35 - exclude DO columns (100% null)
    print("=== LSWW35 ===")
    for split in ['train', 'val', 'test']:
        input_file = f'data/lsww_35c_split/{split}.csv'
        output_file = f'data/lsww_35c_split/{split}_clean.csv'

        if os.path.exists(input_file):
            fix_file(input_file, output_file, exclude_cols=['DO-RT', 'DO-PPL1', 'DO-PPL2'])
        else:
            print(f"Warning: {input_file} not found")

    # CAWW35 - keep all columns (DO has valid data)
    print("=== CAWW35 ===")
    for split in ['train', 'val', 'test']:
        input_file = f'data/caww_35c_split/{split}.csv'
        output_file = f'data/caww_35c_split/{split}_clean.csv'

        if os.path.exists(input_file):
            fix_file(input_file, output_file, exclude_cols=None)
        else:
            print(f"Warning: {input_file} not found")

    print("="*60)
    print("All data processing completed!")
    print("="*60)

if __name__ == "__main__":
    main()
