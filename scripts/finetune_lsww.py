#!/usr/bin/env python3
"""
LSWW数据集的Finetuning:
1. 使用CAWW29 Transformer最佳超参数训练LSWW29模型
2. 对LSWW35进行三种finetuning
"""

import json
import sys
from pathlib import Path
import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.transformer_regressor import TransformerRegressor
from scripts.utils import (
    build_dataset_bundle,
    compute_scalers,
    get_feature_and_target_indices,
    load_time_series,
    scale_values,
)


def load_data(csv_path: Path, input_columns: list, output_columns: list, history_length: int):
    """Load and prepare data - directly load CSV without dropping nulls"""
    df = pl.read_csv(csv_path, encoding="utf-8-sig")
    columns = list(df.columns)

    # Get only the columns we need
    model_columns = input_columns + output_columns
    df = df.select(model_columns)

    # Handle any null values - fill with forward fill then backward fill
    df = df.fill_null(strategy="forward")
    df = df.fill_null(strategy="backward")

    # Convert to numpy
    values = df.to_numpy().astype(np.float32)

    # Get indices
    feature_indices = list(range(len(input_columns)))
    target_indices = list(range(len(input_columns), len(input_columns) + len(output_columns)))

    return values, feature_indices, target_indices


def train_transformer(
    train_csv: Path, val_csv: Path,
    input_columns: list, output_columns: list,
    output_dir: Path, history_length: int = 32,
    d_model: int = 240, nhead: int = 4, num_encoder_layers: int = 4,
    dim_feedforward: int = 429, dropout: float = 0.063,
    max_epochs: int = 150, batch_size: int = 65, learning_rate: float = 0.00168,
    patience: int = 15, seed: int = 42
):
    """Train Transformer on LSWW29 data"""
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Load data
    train_vals, f_idx, t_idx = load_data(train_csv, input_columns, output_columns, history_length)
    val_vals, _, _ = load_data(val_csv, input_columns, output_columns, history_length)

    scalers_mean, scalers_std = compute_scalers(train_vals)
    scaled_train = scale_values(train_vals, scalers_mean, scalers_std)
    scaled_val = scale_values(val_vals, scalers_mean, scalers_std)

    train_bundle = build_dataset_bundle("TRANSFORMER", scaled_train, f_idx, t_idx, history_length)
    val_bundle = build_dataset_bundle("TRANSFORMER", scaled_val, f_idx, t_idx, history_length)

    train_loader = DataLoader(train_bundle.dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_bundle.dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = TransformerRegressor(
        input_dim=train_bundle.input_dim,
        output_dim=train_bundle.targets.shape[1],
        d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward, dropout=dropout
    ).to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss, n_samples = 0, 0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * features.size(0)
            n_samples += features.size(0)
        train_loss = total_loss / n_samples

        model.eval()
        total_loss, n_samples = 0, 0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * features.size(0)
                n_samples += features.size(0)
        val_loss = total_loss / n_samples

        print(f"Epoch {epoch}: train={train_loss:.6f}, val={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            np.savez(output_dir / "scalers.npz", mean=scalers_mean, std=scalers_std)
        else:
            if epoch - best_epoch >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return model, best_val_loss, best_epoch


def finetune_transformer(
    base_model_path: Path,
    train_csv: Path, val_csv: Path, test_csv: Path,
    input_columns: list, output_columns: list,
    output_dir: Path, history_length: int = 32,
    mode: str = "full",  # full, partial, adapter
    max_epochs: int = 50, batch_size: int = 128, learning_rate: float = 5e-5,
    patience: int = 8, seed: int = 42
):
    """Finetune Transformer with different modes"""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_vals, f_idx, t_idx = load_data(train_csv, input_columns, output_columns, history_length)
    val_vals, _, _ = load_data(val_csv, input_columns, output_columns, history_length)
    test_vals, _, _ = load_data(test_csv, input_columns, output_columns, history_length)

    scalers_mean, scalers_std = compute_scalers(train_vals)
    scaled_train = scale_values(train_vals, scalers_mean, scalers_std)
    scaled_val = scale_values(val_vals, scalers_mean, scalers_std)
    scaled_test = scale_values(test_vals, scalers_mean, scalers_std)

    train_bundle = build_dataset_bundle("TRANSFORMER", scaled_train, f_idx, t_idx, history_length)
    val_bundle = build_dataset_bundle("TRANSFORMER", scaled_val, f_idx, t_idx, history_length)
    test_bundle = build_dataset_bundle("TRANSFORMER", scaled_test, f_idx, t_idx, history_length)

    train_loader = DataLoader(train_bundle.dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_bundle.dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_bundle.dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = TransformerRegressor(
        input_dim=train_bundle.input_dim,
        output_dim=train_bundle.targets.shape[1],
        d_model=240, nhead=4, num_encoder_layers=4,
        dim_feedforward=429, dropout=0.063
    )
    model.load_state_dict(torch.load(base_model_path))

    # Configure finetuning mode
    if mode == "full":
        for param in model.parameters():
            param.requires_grad = True
        lr = learning_rate
    elif mode == "partial":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True
        lr = 1e-3
    else:  # adapter
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True
        for param in model.input_projection.parameters():
            param.requires_grad = True
        lr = 1e-3

    model = model.to(device)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss, n_samples = 0, 0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * features.size(0)
            n_samples += features.size(0)
        train_loss = total_loss / n_samples

        model.eval()
        total_loss, n_samples = 0, 0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * features.size(0)
                n_samples += features.size(0)
        val_loss = total_loss / n_samples

        print(f"Epoch {epoch}: train={train_loss:.6f}, val={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Evaluate on test
    model.load_state_dict(torch.load(output_dir / "best_model.pt"))
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Inverse transform
    target_mean = scalers_mean[t_idx]
    target_std = scalers_std[t_idx]
    all_preds_original = all_preds * target_std + target_mean
    all_targets_original = all_targets * target_std + target_mean

    mse = np.mean((all_preds_original - all_targets_original) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_preds_original - all_targets_original))
    r2 = 1 - np.sum((all_targets_original - all_preds_original) ** 2) / np.sum((all_targets_original - np.mean(all_targets_original)) ** 2)

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
    }


def main():
    print("="*60)
    print("LSWW Finetuning Experiment")
    print("="*60)

    # LSWW data paths
    lsww_29_train = REPO_ROOT / "data/lsww_29c_split/train.csv"
    lsww_29_val = REPO_ROOT / "data/lsww_29c_split/val.csv"
    lsww_35_train = REPO_ROOT / "data/lsww_35c_split/train.csv"
    lsww_35_val = REPO_ROOT / "data/lsww_35c_split/val.csv"
    lsww_35_test = REPO_ROOT / "data/lsww_35c_split/test.csv"

    # Same columns as CAWW (but DO columns are missing in LSWW data)
    input_cols = ["TRC-DT", "pH-DT", "cond-DT", "TRC-RT", "pH-RT", "cond-RT", "fDOM-RT", "TOC-RT", "DOC-RT"]
    output_cols = ["TRC-PPL1", "TRC-PPL2", "pH-PPL1", "pH-PPL2", "cond-PPL1", "cond-PPL2", "TOC-PPL1", "TOC-PPL2"]

    results = {}

    # Step 1: Train on LSWW29 using CAWW29 Transformer best params
    print("\n" + "="*60)
    print("Step 1: Training LSWW29 Transformer")
    print("="*60)

    lsww29_model_dir = REPO_ROOT / "outputs/lsww29_transformer"
    lsww29_model_dir.mkdir(parents=True, exist_ok=True)

    model, best_val, best_epoch = train_transformer(
        lsww_29_train, lsww_29_val,
        input_cols, output_cols,
        lsww29_model_dir,
        history_length=32,
        d_model=240, nhead=4, num_encoder_layers=4,
        dim_feedforward=429, dropout=0.063,
        max_epochs=150, batch_size=128, learning_rate=0.00168,
        patience=15
    )
    print(f"LSWW29 training completed: best_val_loss={best_val:.6f}, best_epoch={best_epoch}")
    results["lsww29_training"] = {"best_val_loss": float(best_val), "best_epoch": best_epoch}

    # Step 2: Finetune on LSWW35
    print("\n" + "="*60)
    print("Step 2: LSWW35 Finetuning")
    print("="*60)

    # Full finetuning
    print("\n--- Full Finetuning ---")
    full_dir = REPO_ROOT / "outputs/finetune/lsww35_full"
    full_dir.mkdir(parents=True, exist_ok=True)
    results["lsww35_full"] = finetune_transformer(
        lsww29_model_dir / "best_model.pt",
        lsww_35_train, lsww_35_val, lsww_35_test,
        input_cols, output_cols, full_dir,
        mode="full", learning_rate=5e-5
    )

    # Partial finetuning
    print("\n--- Partial Finetuning ---")
    partial_dir = REPO_ROOT / "outputs/finetune/lsww35_partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    results["lsww35_partial"] = finetune_transformer(
        lsww29_model_dir / "best_model.pt",
        lsww_35_train, lsww_35_val, lsww_35_test,
        input_cols, output_cols, partial_dir,
        mode="partial"
    )

    # Adapter finetuning
    print("\n--- Adapter Finetuning ---")
    adapter_dir = REPO_ROOT / "outputs/finetune/lsww35_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    results["lsww35_adapter"] = finetune_transformer(
        lsww29_model_dir / "best_model.pt",
        lsww_35_train, lsww_35_val, lsww_35_test,
        input_cols, output_cols, adapter_dir,
        mode="adapter"
    )

    # Save results
    output_file = REPO_ROOT / "outputs/finetune_results.json"
    with open(output_file, 'r') as f:
        existing = json.load(f)

    # Add LSWW results
    existing["lsww29_training"] = results["lsww29_training"]
    existing["lsww35_full"] = results["lsww35_full"]
    existing["lsww35_partial"] = results["lsww35_partial"]
    existing["lsww35_adapter"] = results["lsww35_adapter"]

    with open(output_file, 'w') as f:
        json.dump(existing, f, indent=2)

    print("\n" + "="*60)
    print("Finetuning Results Summary")
    print("="*60)
    print(f"CAWW35 Full: R2={existing['caww35_full']['r2']:.4f}, RMSE={existing['caww35_full']['rmse']:.4f}")
    print(f"CAWW35 Partial: R2={existing['caww35_partial']['r2']:.4f}, RMSE={existing['caww35_partial']['rmse']:.4f}")
    print(f"CAWW35 Adapter: R2={existing['caww35_lora']['r2']:.4f}, RMSE={existing['caww35_lora']['rmse']:.4f}")
    print(f"LSWW35 Full: R2={results['lsww35_full']['r2']:.4f}, RMSE={results['lsww35_full']['rmse']:.4f}")
    print(f"LSWW35 Partial: R2={results['lsww35_partial']['r2']:.4f}, RMSE={results['lsww35_partial']['rmse']:.4f}")
    print(f"LSWW35 Adapter: R2={results['lsww35_adapter']['r2']:.4f}, RMSE={results['lsww35_adapter']['rmse']:.4f}")


if __name__ == "__main__":
    main()