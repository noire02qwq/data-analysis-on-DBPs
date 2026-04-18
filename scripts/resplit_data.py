#!/usr/bin/env python3
"""Re-split data with random shuffle to fix distribution shift issue."""

import pandas as pd
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

def main():
    # Load original data
    print("Loading data...")
    df = pd.read_csv(DATA_DIR / "imputed_data.csv", encoding="utf-8-sig")
    print(f"Original shape: {df.shape}")

    # Get timestamp column
    ts_col = "Date, Time"

    # Define columns
    input_cols = ["TRC-DT", "pH-DT", "cond-DT", "TRC-RT", "pH-RT", "cond-RT", "fDOM-RT", "DO-RT", "TOC-RT", "DOC-RT"]
    output_cols = ["TRC-PPL1", "TRC-PPL2", "pH-PPL1", "pH-PPL2", "cond-PPL1", "cond-PPL2", "TOC-PPL1", "TOC-PPL2"]
    keep_cols = [ts_col] + input_cols + output_cols

    # Keep only relevant columns
    cols_to_keep = [c for c in keep_cols if c in df.columns]
    df = df[cols_to_keep]

    # Drop rows with NaN in any column
    df_clean = df.dropna()
    print(f"After dropping NaN: {df_clean.shape}")

    # Shuffle
    print("Shuffling data...")
    df_clean = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split: 70:15:15
    n = len(df_clean)
    train_size = int(n * 0.7)
    val_size = int(n * 0.15)

    train_df = df_clean.iloc[:train_size]
    val_df = df_clean.iloc[train_size:train_size + val_size]
    test_df = df_clean.iloc[train_size + val_size:]

    print(f"New split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

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