#!/usr/bin/env python3
"""Re-split data maintaining temporal order for time series."""

import pandas as pd
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

def main():
    print("Loading imputed data...")
    df = pd.read_csv(DATA_DIR / "imputed_data.csv", encoding="utf-8-sig")
    print(f"Original shape: {df.shape}")

    # Parse timestamp
    df['_ts'] = pd.to_datetime(df['Date, Time'], format='%Y/%m/%d %H:%M')
    df = df.sort_values('_ts').reset_index(drop=True)

    # Define columns
    input_cols = ["TRC-DT", "pH-DT", "cond-DT", "TRC-RT", "pH-RT", "cond-RT", "fDOM-RT", "DO-RT", "TOC-RT", "DOC-RT"]
    output_cols = ["TRC-PPL1", "TRC-PPL2", "pH-PPL1", "pH-PPL2", "cond-PPL1", "cond-PPL2", "TOC-PPL1", "TOC-PPL2"]

    # Keep only needed columns
    keep_cols = ["Date, Time"] + input_cols + output_cols
    df = df[keep_cols].dropna()

    print(f"After cleaning: {df.shape}")

    # Temporal split: 70:15:15
    n = len(df)
    train_size = int(n * 0.7)
    val_size = int(n * 0.15)

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]

    print(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")


    # Save
    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    val_df.to_csv(DATA_DIR / "val.csv", index=False)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)
    print("Done!")

    # Verify
    print("\nVerification - TRC-PPL1 distribution:")
    print(f"  Train: {train_df['TRC-PPL1'].min():.4f} - {train_df['TRC-PPL1'].max():.4f}, mean={train_df['TRC-PPL1'].mean():.4f}")
    print(f"  Val:   {val_df['TRC-PPL1'].min():.4f} - {val_df['TRC-PPL1'].max():.4f}, mean={val_df['TRC-PPL1'].mean():.4f}")
    print(f"  Test:  {test_df['TRC-PPL1'].min():.4f} - {test_df['TRC-PPL1'].max():.4f}, mean={test_df['TRC-PPL1'].mean():.4f}")

if __name__ == "__main__":
    main()
