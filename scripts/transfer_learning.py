#!/usr/bin/env python3
"""
Transfer Learning Pipeline for DBPs Regression Models.

Tasks:
1. LSWW29 New Training - train from scratch using CAWW29 best hyperparameters
2. CAWW35 Fine-tuning - 3 modes (full, partial, frozen) from CAWW29 model
3. LSWW35 Fine-tuning - 3 modes (full, partial, frozen) from LSWW29 model

Output Structure:
outputs/transfer_learning/
├── lsw29_new/                    # Task 1: New training on LSWW29
│   ├── best_config.toml
│   ├── final_model/
│   └── test_results/
├── caww35_full/                  # Task 2a: Full fine-tuning
├── caww35_partial/               # Task 2b: Partial fine-tuning
├── caww35_frozen/                # Task 2c: Frozen feature extractor
├── lsw35_full/                   # Task 3a: Full fine-tuning
├── lsw35_partial/                # Task 3b: Partial fine-tuning
└── lsw35_frozen/                 # Task 3c: Frozen feature extractor
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import tomllib as tomli
except ImportError:
    import tomli

try:
    import tomli_w
except ImportError:
    tomli_w = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import TransformerRegressor
from scripts.utils import (
    DatasetBundle,
    build_dataset_bundle,
    compute_scalers,
    get_feature_and_target_indices,
    load_time_series,
    scale_values,
)


def load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as fh:
        return tomli.load(fh)


def save_toml(path: Path, data: Dict[str, Any]) -> None:
    with path.open("wb") as fh:
        tomli_w.dump(data, fh)


def get_best_transformer_config() -> Dict[str, Any]:
    """Get best Transformer config from CAWW29 experiment."""
    best_config_path = REPO_ROOT / "outputs" / "temporal_experiment" / "models" / "transformer" / "best_config.toml"

    if best_config_path.exists():
        return load_toml(best_config_path)

    # Fallback: use default config
    default_config_path = REPO_ROOT / "models" / "configs" / "transformer_config.toml"
    return load_toml(default_config_path)


def prepare_data(
    data_dir: Path,
    input_columns: List[str],
    output_columns: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and prepare train/val/test data."""
    import polars as pl

    # Load CSV files
    train_df = pl.read_csv(data_dir / "train.csv", encoding="utf-8-sig")
    val_df = pl.read_csv(data_dir / "val.csv", encoding="utf-8-sig")
    test_df = pl.read_csv(data_dir / "test.csv", encoding="utf-8-sig")

    # Handle timestamp column
    if "Date, Time" in train_df.columns:
        train_df = load_time_series(data_dir / "train.csv", "Date, Time")
        val_df = load_time_series(data_dir / "val.csv", "Date, Time")
        test_df = load_time_series(data_dir / "test.csv", "Date, Time")

    # Select columns
    all_columns = input_columns + output_columns
    train_df = train_df.select(all_columns)
    val_df = val_df.select(all_columns)
    test_df = test_df.select(all_columns)

    # Convert to numpy
    train_values = train_df.to_numpy().astype(np.float32)
    val_values = val_df.to_numpy().astype(np.float32)
    test_values = test_df.to_numpy().astype(np.float32)

    # Compute scalers from training data
    scalers_mean, scalers_std = compute_scalers(train_values)

    # Scale all data
    train_scaled = scale_values(train_values, scalers_mean, scalers_std)
    val_scaled = scale_values(val_values, scalers_mean, scalers_std)
    test_scaled = scale_values(test_values, scalers_mean, scalers_std)

    # Get feature/target indices
    feature_indices, target_indices = get_feature_and_target_indices(
        all_columns, input_columns, output_columns
    )

    return (
        train_scaled, val_scaled, test_scaled,
        scalers_mean, scalers_std,
        np.array(feature_indices), np.array(target_indices),
        np.array(all_columns)
    )


def create_transformer_model(config: Dict[str, Any], input_dim: int, output_dim: int) -> TransformerRegressor:
    """Create Transformer model from config."""
    model_params = config.get("model", {})

    return TransformerRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        d_model=model_params.get("d_model", 128),
        nhead=model_params.get("nhead", 8),
        num_encoder_layers=model_params.get("num_encoder_layers", 4),
        dim_feedforward=model_params.get("dim_feedforward", 512),
        dropout=model_params.get("dropout", 0.1),
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Train model with given configuration."""
    training_params = config.get("training", {})

    max_epochs = training_params.get("max_epochs", 100)
    learning_rate = training_params.get("learning_rate", 0.001)
    weight_decay = training_params.get("weight_decay", 0.0)
    patience = training_params.get("patience", 10)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(max_epochs):
        # Training
        model.train()
        train_losses = []
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # Save best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    torch.save(model.state_dict(), output_dir / "best_model.pt")

    # Save history
    import csv
    with open(output_dir / "loss_history.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i in range(len(history["epoch"])):
            writer.writerow([history["epoch"][i], history["train_loss"][i], history["val_loss"][i]])

    return {
        "best_val_loss": best_val_loss,
        "epochs_trained": len(history["epoch"]),
    }


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    scalers_mean: np.ndarray,
    scalers_std: np.ndarray,
    target_indices: np.ndarray,
    output_columns: List[str],
    output_dir: Path,
    device: torch.device,
) -> Dict[str, float]:
    """Test model and compute metrics."""
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            outputs = model(features)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Inverse transform
    target_mean = scalers_mean[target_indices]
    target_std = scalers_std[target_indices]
    predictions_orig = predictions * target_std + target_mean
    targets_orig = targets * target_std + target_mean

    # Compute metrics
    mse = np.mean((predictions_orig - targets_orig) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions_orig - targets_orig))

    ss_res = np.sum((targets_orig - predictions_orig) ** 2)
    ss_tot = np.sum((targets_orig - np.mean(targets_orig, axis=0)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Save predictions
    import polars as pl
    pred_df = pl.DataFrame(predictions_orig, schema=output_columns)
    pred_df.write_csv(output_dir / "test_predictions.csv")

    # Save metrics
    metrics = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }

    with open(output_dir / "test_metrics.csv", 'w') as f:
        f.write("metric,value\n")
        for k, v in metrics.items():
            f.write(f"{k},{v}\n")

    return metrics


def task1_lsw29_new_training(
    output_dir: Path,
    device: torch.device,
) -> bool:
    """Task 1: LSWW29 new training using CAWW29 best hyperparameters."""
    print("\n" + "="*80)
    print("Task 1: LSWW29 New Training")
    print("="*80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get best Transformer config from CAWW29
    best_config = get_best_transformer_config()

    # Update data paths for LSWW29
    best_config["data"]["train_csv"] = str(REPO_ROOT / "data" / "lsww_29c_split" / "train.csv")
    best_config["data"]["val_csv"] = str(REPO_ROOT / "data" / "lsww_29c_split" / "val.csv")
    best_config["data"]["test_csv"] = str(REPO_ROOT / "data" / "lsww_29c_split" / "test.csv")

    # Save config
    save_toml(output_dir / "best_config.toml", best_config)

    # Prepare data
    data_result = prepare_data(
        REPO_ROOT / "data" / "lsww_29c_split",
        best_config["data"]["input_columns"],
        best_config["data"]["output_columns"],
    )

    if data_result is None:
        print("Failed to prepare data")
        return False

    (train_scaled, val_scaled, test_scaled,
     scalers_mean, scalers_std,
     feature_indices, target_indices,
     all_columns) = data_result

    # Create datasets
    history_length = best_config.get("model", {}).get("history_length", 1)

    train_bundle = build_dataset_bundle(
        "TRANSFORMER", train_scaled, feature_indices, target_indices, history_length
    )
    val_bundle = build_dataset_bundle(
        "TRANSFORMER", val_scaled, feature_indices, target_indices, history_length
    )
    test_bundle = build_dataset_bundle(
        "TRANSFORMER", test_scaled, feature_indices, target_indices, history_length
    )

    # Create data loaders
    batch_size = best_config.get("training", {}).get("batch_size", 128)
    train_loader = DataLoader(train_bundle.dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_bundle.dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_bundle.dataset, batch_size=batch_size, shuffle=False)

    # Create model
    input_dim = train_bundle.input_dim
    output_dim = len(best_config["data"]["output_columns"])

    model = create_transformer_model(best_config, input_dim, output_dim)
    model = model.to(device)

    # Train
    final_model_dir = output_dir / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)

    train_result = train_model(
        model, train_loader, val_loader, best_config,
        final_model_dir, device
    )

    print(f"Training completed. Best val loss: {train_result['best_val_loss']:.6f}")

    # Test
    test_output_dir = output_dir / "test_results"
    test_output_dir.mkdir(parents=True, exist_ok=True)

    # Load best model
    model.load_state_dict(torch.load(final_model_dir / "best_model.pt", map_location=device))

    test_metrics = test_model(
        model, test_loader, scalers_mean, scalers_std,
        target_indices, best_config["data"]["output_columns"],
        test_output_dir, device
    )

    print(f"Test metrics: RMSE={test_metrics['rmse']:.6f}, MAE={test_metrics['mae']:.6f}, R2={test_metrics['r2']:.6f}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Transfer Learning Pipeline")
    parser.add_argument("--task", choices=["lsw29", "caww35", "lsw35", "all"], default="all",
                       help="Which task to run")
    parser.add_argument("--mode", choices=["full", "partial", "frozen", "all"], default="all",
                       help="Fine-tuning mode (for caww35/lsw35 tasks)")
    parser.add_argument("--output-dir", default="outputs/transfer_learning",
                       help="Output directory")
    parser.add_argument("--device", default="auto", help="Device to use (auto/cpu/cuda)")
    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Run tasks
    results = {}

    if args.task in ["lsw29", "all"]:
        success = task1_lsw29_new_training(output_base / "lsw29_new", device)
        results["lsw29_new"] = "success" if success else "failed"

    print("\n" + "="*80)
    print("Transfer Learning Pipeline Complete")
    print("="*80)
    print("\nResults:")
    for task, status in results.items():
        print(f"  {task}: {status}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
