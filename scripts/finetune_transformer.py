#!/usr/bin/env python3
"""
Fine-tune pre-trained Transformer model on new datasets.
Implements two fine-tuning methods:
1. Full fine-tuning: update all parameters with smaller learning rate
2. Partial fine-tuning: freeze encoder, only update final regression head
"""

from __future__ import annotations

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Transformer on new dataset"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["caww_35c", "lsww_29c", "lsww_35c"],
        help="Dataset to fine-tune on",
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=["full", "partial"],
        help="Fine-tuning method: full (all params) or partial (head only)",
    )
    parser.add_argument(
        "--pretrained-model",
        default="outputs/transformer_final/best_model.pt",
        help="Path to pre-trained Transformer model",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing dataset",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/finetune",
        help="Output directory",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning (typically smaller than training)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def load_pretrained_model(
    model_path: str,
    model_config: Dict[str, Any],
    device: torch.device,
) -> TransformerRegressor:
    """Load pre-trained model from checkpoint."""
    model = TransformerRegressor(**model_config)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    return model


def freeze_encoder(model: TransformerRegressor):
    """Freeze all layers except the final regression head."""
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the final fully connected layers
    # The head is the last layer in the model
    if hasattr(model, 'fc'):
        for param in model.fc.parameters():
            param.requires_grad = True
    elif hasattr(model, 'final_layer'):
        for param in model.final_layer.parameters():
            param.requires_grad = True
    else:
        # For our architecture, the last part is the output projection
        # Find the last linear layer and unfreeze
        for name, param in model.named_parameters():
            if 'fc' in name or 'head' in name or 'output' in name:
                param.requires_grad = True

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Partial fine-tuning: {trainable_params:,}/{total_params:,} parameters trainable")


def train_epoch(
    model: TransformerRegressor,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate(
    model: TransformerRegressor,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            loss = criterion(y_pred, y)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def test(
    model: TransformerRegressor,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Test the model."""
    return validate(model, test_loader, criterion, device)


def plot_training_curve(
    epochs: list[int],
    train_losses: list[float],
    val_losses: list[float],
    output_path: Path,
):
    """Plot training curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Transformer Fine-Tuning - Training Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_loss_history(output_path: Path, epochs: list[int], train_losses: list[float], val_losses: list[float]):
    """Save loss history to CSV."""
    with open(output_path, 'w') as f:
        f.write('epoch,train_loss,val_loss\n')
        for e, tr, va in zip(epochs, train_losses, val_losses):
            f.write(f'{e},{tr:.8f},{va:.8f}\n')


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset paths
    dataset_name = args.dataset
    data_dir = Path(args.data_dir)
    imputed_path = data_dir / f"{dataset_name}_imputed_data.csv"

    if not imputed_path.exists():
        print(f"Error: Imputed data not found at {imputed_path}")
        sys.exit(1)

    # Get split paths
    split_dir = data_dir / f"{dataset_name}_split"
    train_path = split_dir / "train.csv"
    val_path = split_dir / "val.csv"
    test_path = split_dir / "test.csv"

    # Candidate columns - same as original, we'll filter what actually exists
    candidate_input_columns = [
        "minutes_since_start",
        "TRC-DT", "pH-DT", "cond-DT",
        "TRC-RT", "pH-RT", "fDOM-RT", "DO-RT",
        "TOC-RT", "DOC-RT",
    ]
    candidate_output_columns = [
        "TRC-PPL1", "TRC-PPL2", "pH-PPL1", "pH-PPL2",
        "cond-PPL1", "cond-PPL2", "TOC-PPL1", "TOC-PPL2",
        "DOC-PPL1", "DOC-PPL2",
    ]

    # Read first row to check which columns exist + check for timestamp column
    df_temp = pl.read_csv(train_path, n_rows=1)
    ts_col = "Date, Time" if "Date, Time" in df_temp.columns else None
    all_available_cols = set(df_temp.columns)
    input_columns = [c for c in candidate_input_columns if c in all_available_cols]
    output_columns = [c for c in candidate_output_columns if c in all_available_cols]

    # Custom loading that avoids dropping all rows due to missing values in unused columns
    # The standard load_time_series drops all rows because it drops any row with any null anywhere
    # But we don't care about nulls in columns we don't use, so we filter first then drop nulls
    # We also drop columns that are completely null (all values missing) from the original data
    def load_custom(csv_path: Path, ts_col: Optional[str], keep_cols: list[str]) -> pl.DataFrame:
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
        # Drop any columns that are completely null (entire dataset missing this column)
        # This avoids dropping all rows just because one column is entirely missing
        for col in df.columns:
            if df[col].is_null().all():
                print(f"  Dropping completely null column: {col}")
                df = df.drop(col)
        # Now drop rows that still have any null in the remaining columns
        df = df.drop_nulls()
        return df

    # Load with custom approach
    keep_columns = (["minutes_since_start"] + input_columns + output_columns) if ts_col is not None else (input_columns + output_columns)
    train_df = load_custom(train_path, ts_col, keep_columns)
    val_df = load_custom(val_path, ts_col, keep_columns)
    test_df = load_custom(test_path, ts_col, keep_columns)

    # Update input/output columns to reflect what's actually left after dropping completely null columns
    remaining_cols = set(train_df.columns)
    input_columns = [c for c in input_columns if c in remaining_cols]
    output_columns = [c for c in output_columns if c in remaining_cols]

    print(f"Input columns after removing completely null: {input_columns}")
    print(f"Output columns after removing completely null: {output_columns}")
    print(f"Dataset sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Get feature and target indices
    all_columns = train_df.columns
    feature_indices, target_indices = get_feature_and_target_indices(all_columns, input_columns, output_columns)

    # Convert to numpy arrays
    values_train = train_df.to_numpy().astype(np.float32)
    values_val = val_df.to_numpy().astype(np.float32)
    values_test = test_df.to_numpy().astype(np.float32)

    # Compute scalers from training data
    mean_input, std_input = compute_scalers(values_train[:, feature_indices])
    mean_output, std_output = compute_scalers(values_train[:, target_indices])

    # Scale all splits
    values_train_scaled = values_train.copy()
    values_train_scaled[:, feature_indices] = scale_values(values_train[:, feature_indices], mean_input, std_input)
    values_train_scaled[:, target_indices] = scale_values(values_train[:, target_indices], mean_output, std_output)

    values_val_scaled = values_val.copy()
    values_val_scaled[:, feature_indices] = scale_values(values_val[:, feature_indices], mean_input, std_input)
    values_val_scaled[:, target_indices] = scale_values(values_val[:, target_indices], mean_output, std_output)

    values_test_scaled = values_test.copy()
    values_test_scaled[:, feature_indices] = scale_values(values_test[:, feature_indices], mean_input, std_input)
    values_test_scaled[:, target_indices] = scale_values(values_test[:, target_indices], mean_output, std_output)

    # Build dataset bundle - Transformer is sequence model, need history
    # Original best was 64 based on Bayesian optimization
    history_length = 64
    n_outputs = len(output_columns)

    train_bundle = build_dataset_bundle(
        "TRANSFORMER",
        values_train_scaled,
        feature_indices,
        target_indices,
        history_length,
    )
    val_bundle = build_dataset_bundle(
        "TRANSFORMER",
        values_val_scaled,
        feature_indices,
        target_indices,
        history_length,
    )
    test_bundle = build_dataset_bundle(
        "TRANSFORMER",
        values_test_scaled,
        feature_indices,
        target_indices,
        history_length,
    )

    train_loader = DataLoader(train_bundle.dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_bundle.dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_bundle.dataset, batch_size=args.batch_size, shuffle=False)

    # Model configuration - use actual hyperparameters from original pre-trained model
    # Read from the original config to ensure matching architecture
    original_config_path = Path(args.pretrained_model).parent / "config.toml"
    if original_config_path.exists():
        import tomli
        with open(original_config_path, "rb") as f:
            original_config = tomli.load(f)
        model_cfg = original_config["model"]
        model_config = {
            "input_dim": len(input_columns),
            "output_dim": n_outputs,
            "d_model": model_cfg["d_model"],
            "nhead": model_cfg["nhead"],
            "num_encoder_layers": model_cfg["num_encoder_layers"],
            "dim_feedforward": model_cfg["dim_feedforward"],
            "dropout": model_cfg["dropout"],
        }
    else:
        # Fallback to known best parameters
        model_config = {
            "input_dim": len(input_columns),
            "output_dim": n_outputs,
            "d_model": 176,
            "nhead": 4,
            "num_encoder_layers": 4,
            "dim_feedforward": 919,
            "dropout": 0.21,
        }

    # Load pre-trained model and handle dimension mismatches
    print(f"Loading pre-trained model from: {args.pretrained_model}")
    # Create model with current input and output dimensions for the new dataset
    model = TransformerRegressor(**model_config)
    # Load checkpoint
    checkpoint = torch.load(args.pretrained_model, map_location=device)

    # Handle mismatches:
    # 1. Head: output dimension changed because new dataset has different number of outputs
    # 2. Input projection: input dimension may have changed if some input columns are completely missing in new dataset
    filtered_checkpoint = {}
    for k, v in checkpoint.items():
        if k.startswith('head.'):
            continue  # Will create new head for output dimension
        if k == 'input_projection.weight' and v.shape[1] != model_config['input_dim']:
            # Input dimension mismatch - original had more inputs, we copy the weights for the columns we keep
            print(f"Handling input dimension mismatch: original {v.shape[1]} inputs → new {model_config['input_dim']}")
            new_weight = model.input_projection.weight.clone()
            # Copy the weights for columns that exist in both
            # We assume the column order matches for the common columns
            new_weight[:, :] = v[:, :model_config['input_dim']]
            filtered_checkpoint[k] = new_weight
        elif k == 'input_projection.bias' and v.shape[0] != model_config['d_model']:
            filtered_checkpoint[k] = v  # bias dimension is d_model, which shouldn't change
        else:
            filtered_checkpoint[k] = v

    missing_keys = model.load_state_dict(filtered_checkpoint, strict=False)

    # The head should be missing, which is expected
    print(f"Loaded pretrained weights (adjusted for input/output dimensions): {len(filtered_checkpoint)} keys loaded")
    if len(missing_keys.missing_keys) > 0:
        print(f"Missing keys (expected for new head): {[k for k in missing_keys.missing_keys if k.startswith('head.')]}")

    model.to(device)

    # Apply fine-tuning method
    if args.method == "partial":
        freeze_encoder(model)

    # Setup training
    criterion = nn.MSELoss()
    if args.method == "full":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        print(f"Full fine-tuning: All {sum(p.numel() for p in model.parameters()):,} parameters trainable")
    else:
        # Only optimize trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=args.learning_rate, weight_decay=1e-5)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.dataset}_{args.method}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Training loop with early stopping
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    epochs_list = []

    for epoch in range(1, args.max_epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss = validate(model, val_loader, criterion, device)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        epochs_list.append(epoch)

        print(f"Epoch {epoch:3d}: train_loss={tr_loss:.6f}, val_loss={va_loss:.6f}")

        # Save best model
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (best at epoch {best_epoch})")
                break

    # Load best model for testing
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    test_loss = test(model, test_loader, criterion, device)
    print(f"\nBest model from epoch {best_epoch}:")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print(f"  Test loss: {test_loss:.6f}")

    # Save last model
    torch.save(model.state_dict(), output_dir / "last_model.pt")

    # Save loss history
    save_loss_history(output_dir / "loss_history.csv", epochs_list, train_losses, val_losses)

    # Plot training curve
    plot_training_curve(epochs_list, train_losses, val_losses, output_dir / "training_curve.png")

    # Save scalers
    np.savez_compressed(
        output_dir / "scalers.npz",
        mean_input=mean_input,
        std_input=std_input,
        mean_output=mean_output,
        std_output=std_output,
    )

    # Save configuration
    config = {
        "model": model_config,
        "training": {
            "method": args.method,
            "learning_rate": args.learning_rate,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "data": {
            "dataset": args.dataset,
            "train_csv": str(train_path),
            "val_csv": str(val_path),
            "test_csv": str(test_path),
            "input_columns": input_columns,
            "output_columns": output_columns,
        },
        "pretrained": {
            "original_model_path": args.pretrained_model,
        },
    }

    with open(output_dir / "config.toml", "wb") as f:
        tomli_w.dump(config, f)

    # Save result
    result = {
        "model_name": f"transformer_{args.dataset}_{args.method}_finetune",
        "model_type": "TRANSFORMER",
        "finetune_method": args.method,
        "dataset": args.dataset,
        "input_columns": input_columns,
        "output_columns": output_columns,
        "eval": {
            "best_val_loss": best_val_loss,
            "test_loss": test_loss,
            "best_epoch": best_epoch,
        },
    }

    with open(output_dir / "result.toml", "wb") as f:
        tomli_w.dump(result, f)

    # Also save as json for easier processing
    with open(output_dir / "result.json", "w") as f:
        json.dump({
            "best_val_loss": float(best_val_loss),
            "test_loss": float(test_loss),
            "best_epoch": best_epoch,
            "finetune_method": args.method,
            "dataset": args.dataset,
        }, f, indent=2)

    # Create test results directory with predictions
    test_results_dir = output_dir / "test_results"
    test_results_dir.mkdir(exist_ok=True)

    print(f"\nFine-tuning complete!")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
