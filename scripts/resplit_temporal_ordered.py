#!/usr/bin/env python3
"""Re-split data maintaining TEMPORAL ORDER for time series (NO SHUFFLING)."""

import pandas as pd
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

def main():
    print("Loading imputed data...")
    df = pd.read_csv(DATA_DIR / "imputed_data.csv", encoding="utf-8-sig")
    print(f"Original shape: {df.shape}")

    # Parse timestamp and SORT BY TIME (maintain temporal order)
    df['_ts'] = pd.to_datetime(df['Date, Time'], format='%Y/%m/%d %H:%M')
    df = df.sort_values('_ts').reset_index(drop=True)
    print(f"Sorted by time: {df.shape}")

    # Define columns
    input_cols = ["TRC-DT", "pH-DT", "cond-DT", "TRC-RT", "pH-RT", "cond-RT", "fDOM-RT", "DO-RT", "TOC-RT", "DOC-RT"]
    output_cols = ["TRC-PPL1", "TRC-PPL2", "pH-PPL1", "pH-PPL2", "cond-PPL1", "cond-PPL2", "TOC-PPL1", "TOC-PPL2"]
    keep_cols = ["Date, Time"] + input_cols + output_cols

    # Keep only needed columns
    cols_to_keep = [c for c in keep_cols if c in df.columns]
    df = df[cols_to_keep].dropna()
    print(f"After cleaning: {df.shape}")

    # TEMPORAL split: 70:15:15 (NO SHUFFLING - maintain time order)
    n = len(df)
    train_size = int(n * 0.7)
    val_size = int(n * 0.15)

    train_df = df.iloc[:train_size]      # First 70% - earliest times
    val_df = df.iloc[train_size:train_size + val_size]  # Next 15%
    test_df = df.iloc[train_size + val_size:]  # Last 15% - latest times

    print(f"Temporal split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print(f"  Train: {train_df['Date, Time'].iloc[0]} to {train_df['Date, Time'].iloc[-1]}")
    print(f"  Val:   {val_df['Date, Time'].iloc[0]} to {val_df['Date, Time'].iloc[-1]}")
    print(f"  Test:  {test_df['Date, Time'].iloc[0]} to {test_df['Date, Time'].iloc[-1]}")

    # Save
    print("Saving...")
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
