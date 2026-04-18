#!/usr/bin/env python3
"""
Adapter-based finetuning that handles dimension mismatch.
Standalone version with embedded data loading functions.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import tomli
import tomli_w
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models import TransformerRegressor
from scripts.utils import (
    DatasetBundle,
    build_dataset_bundle,
    compute_scalers,
    scale_values,
    get_feature_and_target_indices,
)


class AdapterLayer(nn.Module):
    """Bottleneck adapter layer."""
    def __init__(self, d_model: int, bottleneck_dim: int = 64):
        super().__init__()
        self.down_project = nn.Linear(d_model, bottleneck_dim)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(bottleneck_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return x + residual


class AdaptiveTransformerWithHeads(nn.Module):
    """
    Transformer with:
    1. New input projection for dimension mismatch
    2. Adapter layers after each encoder layer
    3. New output head for dimension mismatch
    Only adapters and new projections are trainable.
    """
    def __init__(
        self,
        base_model: TransformerRegressor,
        new_input_dim: int,
        new_output_dim: int,
        bottleneck_dim: int = 64
    ):
        super().__init__()
        self.d_model = base_model.d_model

        # Freeze the base model
        for param in base_model.parameters():
            param.requires_grad = False
        base_model.eval()
        self.base_model = base_model

        # New input projection (trainable)
        self.new_input_projection = nn.Linear(new_input_dim, self.d_model)

        # Positional encoding (from base model, frozen)
        self.pos_encoding = base_model.pos_encoding

        # Adapter layers (trainable) - one after each encoder layer
        num_layers = len(base_model.transformer_encoder.layers)
        self.adapters = nn.ModuleList([
            AdapterLayer(self.d_model, bottleneck_dim) for _ in range(num_layers)
        ])

        # New output head (trainable)
        self.new_head = nn.Linear(self.d_model, new_output_dim)

        # Count trainable parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # New input projection (trainable)
        x = self.new_input_projection(x)

        # Positional encoding
        x = self.pos_encoding(x)

        # Pass through frozen encoder with trainable adapters
        for i, layer in enumerate(self.base_model.transformer_encoder.layers):
            x = layer(x)
            x = self.adapters[i](x)

        # Take last time step
        x = x[:, -1, :]

        # New output head (trainable)
        x = self.new_head(x)
        return x


def load_custom(csv_path: Path, ts_col: Optional[str], keep_cols: list[str]) -> pl.DataFrame:
    """Custom data loader that handles null columns."""
    df = pl.read_csv(csv_path, encoding="utf-8-sig")
    if ts_col is not None:
        df = df.with_columns([
            pl.col(ts_col).str.to_datetime().alias("_timestamp")
        ])
        df = df.sort("_timestamp")
        min_ts = df["_timestamp"][0]
        df = df.with_columns([
            ((pl.col("_timestamp") - min_ts).dt.total_seconds() / 60.0).alias("minutes_since_start")
        ])
    # Keep only columns we need
    df = df.select(keep_cols)
    # Convert to numeric, coercing errors to null
    numeric_cols = []
    for col in df.columns:
        try:
            numeric_cols.append(pl.col(col).cast(pl.Float64))
        except:
            numeric_cols.append(pl.col(col))
    df = df.with_columns(numeric_cols)
    # Drop any columns that are completely null
    for col in df.columns:
        if df[col].is_null().all():
            print(f"  Dropping completely null column: {col}")
            df = df.drop(col)
    # Now drop rows that still have any null in the remaining columns
    df = df.drop_nulls()
    return df


def run_adapter_finetuning(args):
    """Run adapter finetuning experiment."""
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"Adapter Finetuning: {args.dataset}")
    print(f"{'='*60}")

    # Load pretrained model
    print(f"\nLoading pretrained model from: {args.pretrained_model}")

    # First load the checkpoint to get config
    checkpoint = torch.load(args.pretrained_model, map_location=device)

    # Get original dimensions from checkpoint
    original_input_dim = checkpoint['input_projection.weight'].shape[1]
    original_output_dim = checkpoint['head.0.weight'].shape[0]
    d_model = checkpoint['input_projection.weight'].shape[0]

    print(f"  Original model: {original_input_dim} inputs, {original_output_dim} outputs")
    print(f"  d_model: {d_model}")

    # Determine new dimensions based on dataset
    if args.dataset == "lsww_35c":
        new_input_dim = 8   # DO columns excluded
        new_output_dim = 10
    else:  # caww_35c
        new_input_dim = 9   # All columns including DO
        new_output_dim = 14

    print(f"  Target: {new_input_dim} inputs, {new_output_dim} outputs")

    # Create base model with original dimensions
    base_model = TransformerRegressor(
        input_dim=original_input_dim,
        output_dim=original_output_dim,
        d_model=d_model,
        nhead=8,
        num_encoder_layers=2,
        dim_feedforward=509,
        dropout=0.277,
    ).to(device)

    # Load weights
    base_model.load_state_dict(checkpoint)
    base_model.eval()

    # Create adaptive model
    print("\nCreating adaptive model with:")
    model = AdaptiveTransformerWithHeads(
        base_model=base_model,
        new_input_dim=new_input_dim,
        new_output_dim=new_output_dim,
        bottleneck_dim=args.bottleneck_dim
    ).to(device)

    # Load data
    print("\nLoading data...")

    train_path = Path("data") / f"{args.dataset}_split" / "train_clean.csv"
    val_path = Path("data") / f"{args.dataset}_split" / "val_clean.csv"
    test_path = Path("data") / f"{args.dataset}_split" / "test_clean.csv"

    if args.dataset == "lsww_35c":
        input_columns = ["TRC-DT", "pH-DT", "cond-DT", "TRC-RT", "pH-RT", "fDOM-RT", "TOC-RT", "DOC-RT"]
        output_columns = ["TRC-PPL1", "TRC-PPL2", "pH-PPL1", "pH-PPL2", "cond-PPL1", "cond-PPL2", "TOC-PPL1", "TOC-PPL2", "DOC-PPL1", "DOC-PPL2"]
    else:
        input_columns = ["TRC-DT", "pH-DT", "cond-DT", "TRC-RT", "pH-RT", "fDOM-RT", "DO-RT", "TOC-RT", "DOC-RT"]
        output_columns = ["TRC-PPL1", "TRC-PPL2", "pH-PPL1", "pH-PPL2", "cond-PPL1", "cond-PPL2", "TOC-PPL1", "TOC-PPL2", "DOC-PPL1", "DOC-PPL2", "fDOM-PPL1", "fDOM-PPL2", "DO-PPL1", "DO-PPL2"]

    # Load data using custom loader
    ts_col = None
    keep_cols = input_columns + output_columns

    train_df = load_custom(train_path, ts_col, keep_cols)
    val_df = load_custom(val_path, ts_col, keep_cols)
    test_df = load_custom(test_path, ts_col, keep_cols)

    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Prepare data
    feature_indices, target_indices = get_feature_and_target_indices(
        train_df.columns, input_columns, output_columns
    )

    values_train = train_df.to_numpy().astype(np.float32)
    values_val = val_df.to_numpy().astype(np.float32)
    values_test = test_df.to_numpy().astype(np.float32)

    # Compute scalers
    mean_input, std_input = compute_scalers(values_train[:, feature_indices])
    mean_output, std_output = compute_scalers(values_train[:, target_indices])

    # Scale
    for values in [values_train, values_val, values_test]:
        values[:, feature_indices] = scale_values(values[:, feature_indices], mean_input, std_input)
        values[:, target_indices] = scale_values(values[:, target_indices], mean_output, std_output)

    # Build datasets
    history_length = 64
    train_bundle = build_dataset_bundle("TRANSFORMER", values_train, feature_indices, target_indices, history_length)
    val_bundle = build_dataset_bundle("TRANSFORMER", values_val, feature_indices, target_indices, history_length)
    test_bundle = build_dataset_bundle("TRANSFORMER", values_test, feature_indices, target_indices, history_length)

    train_loader = DataLoader(train_bundle.dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_bundle.dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_bundle.dataset, batch_size=args.batch_size, shuffle=False)

    # Setup training
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate
    )
    criterion = nn.MSELoss()

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting training for {args.max_epochs} epochs...")

    for epoch in range(1, args.max_epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        # Track best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"Epoch {epoch:3d}: train={train_loss:.6f}, val={val_loss:.6f} (Best)")
        else:
            patience_counter += 1
            print(f"Epoch {epoch:3d}: train={train_loss:.6f}, val={val_loss:.6f}")

            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Test
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            test_loss += loss.item()
    test_loss /= len(test_loader)

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Test loss: {test_loss:.6f}")

    # Save results
    result = {
        "model_name": f"transformer_{args.dataset}_adapter_finetune",
        "model_type": "TRANSFORMER",
        "finetune_method": "adapter",
        "dataset": args.dataset,
        "eval": {
            "best_val_loss": float(best_val_loss),
            "test_loss": float(test_loss),
            "best_epoch": best_epoch,
        },
    }

    with open(output_dir / "result.toml", "wb") as f:
        tomli_w.dump(result, f)

    with open(output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    return best_val_loss, test_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["caww_35c", "lsww_35c"])
    parser.add_argument("--pretrained-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bottleneck-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_adapter_finetuning(args)


if __name__ == "__main__":
    main()
