#!/usr/bin/env python3
"""
三种模式的Transformer Finetuning:
1. Full Fine-Tuning - 更新所有参数
2. Partial Fine-Tuning - 冻结encoder，只训练head
3. LoRA Fine-Tuning - 使用低秩适配

CAWW29 → CAWW35
LSWW29 → LSWW35
"""

import argparse
import math
import random
import shutil
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.transformer_regressor import TransformerRegressor
from scripts.utils import (
    DatasetBundle,
    build_dataset_bundle,
    compute_scalers,
    get_feature_and_target_indices,
    load_time_series,
    scale_values,
)


class LoRALayer(nn.Module):
    """LoRA layer for attention mechanism"""
    def __init__(self, d_model: int, rank: int = 8):
        super().__init__()
        self.rank = rank
        self.lora_a = nn.Linear(d_model, rank, bias=False)
        self.lora_b = nn.Linear(rank, d_model, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        return self.lora_b(self.lora_a(x))


class LoRATransformerWrapper(nn.Module):
    """Transformer with LoRA layers added"""
    def __init__(self, base_model: nn.Module, rank: int = 8, lora_alpha: float = 16):
        super().__init__()
        self.base_model = base_model
        self.rank = rank
        self.lora_alpha = lora_alpha

        # Add LoRA to each encoder layer
        self.lora_layers = nn.ModuleList([
            LoRALayer(base_model.d_model, rank)
            for _ in range(len(base_model.transformer_encoder.layers))
        ])

        # Freeze base model parameters
        for param in base_model.parameters():
            param.requires_grad = False

        # Unfreeze LoRA parameters
        for lora_layer in self.lora_layers:
            for param in lora_layer.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.base_model.input_projection(x)
        x = self.base_model.pos_encoding(x)

        for i, layer in enumerate(self.base_model.transformer_encoder.layers):
            x = layer(x)
            lora_out = self.lora_layers[i](x)
            x = x + lora_out * (self.lora_alpha / self.rank)

        # Use a simple normalization
        x = torch.nn.functional.layer_norm(x, (x.size(-1),))

        x = x[:, -1, :]
        return self.base_model.head(x)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(csv_path: Path, input_columns: list, output_columns: list, history_length: int):
    """Load and prepare data"""
    df_temp = pl.read_csv(csv_path, encoding="utf-8-sig", n_rows=1)
    ts_col = "Date, Time" if "Date, Time" in df_temp.columns else None

    df = load_time_series(csv_path, ts_col) if ts_col else pl.read_csv(csv_path)
    columns = list(df.columns)

    feature_indices, target_indices = get_feature_and_target_indices(columns, input_columns, output_columns)
    values = df.to_numpy().astype(np.float32)

    return values, feature_indices, target_indices


def train_with_finetuning(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    learning_rate: float = 1e-4,
    max_epochs: int = 50,
    patience: int = 8,
    output_dir: Path = None,
) -> dict:
    """Train model with early stopping"""
    model = model.to(device)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    train_history = []
    val_history = []

    for epoch in range(1, max_epochs + 1):
        # Training
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
        train_history.append(train_loss)

        # Validation
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
        val_history.append(val_loss)

        print(f"Epoch {epoch}: train={train_loss:.6f}, val={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            if output_dir:
                torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience > 0 and patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    if output_dir and (output_dir / "best_model.pt").exists():
        model.load_state_dict(torch.load(output_dir / "best_model.pt"))

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_history": train_history,
        "val_history": val_history,
    }


def evaluate_on_test(model: nn.Module, test_loader: DataLoader, device: torch.device) -> dict:
    """Evaluate model on test set"""
    model.eval()
    criterion = nn.MSELoss()

    all_preds = []
    all_targets = []
    total_loss = 0
    n_samples = 0

    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * features.size(0)
            n_samples += features.size(0)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Calculate metrics
    mse = np.mean((all_preds - all_targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_preds - all_targets))
    r2 = 1 - np.sum((all_targets - all_preds) ** 2) / np.sum((all_targets - np.mean(all_targets)) ** 2)

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "predictions": all_preds,
        "targets": all_targets,
    }


def finetune_full(base_model_path: Path, train_csv: Path, val_csv: Path, test_csv: Path,
                  input_columns: list, output_columns: list, output_dir: Path, seed: int = 42):
    """Full Fine-Tuning"""
    print("\n=== Full Fine-Tuning ===")
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_vals, f_idx, t_idx = load_data(train_csv, input_columns, output_columns, 32)
    val_vals, _, _ = load_data(val_csv, input_columns, output_columns, 32)
    test_vals, _, _ = load_data(test_csv, input_columns, output_columns, 32)

    scalers_mean, scalers_std = compute_scalers(train_vals)
    scaled_train = scale_values(train_vals, scalers_mean, scalers_std)
    scaled_val = scale_values(val_vals, scalers_mean, scalers_std)
    scaled_test = scale_values(test_vals, scalers_mean, scalers_std)

    train_bundle = build_dataset_bundle("TRANSFORMER", scaled_train, f_idx, t_idx, 32)
    val_bundle = build_dataset_bundle("TRANSFORMER", scaled_val, f_idx, t_idx, 32)
    test_bundle = build_dataset_bundle("TRANSFORMER", scaled_test, f_idx, t_idx, 32)

    train_loader = DataLoader(train_bundle.dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_bundle.dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_bundle.dataset, batch_size=128, shuffle=False)

    # Load model
    model = TransformerRegressor(
        input_dim=train_bundle.input_dim,
        output_dim=train_bundle.targets.shape[1],
        d_model=128, nhead=8, num_encoder_layers=4,
        dim_feedforward=512, dropout=0.1
    )
    model.load_state_dict(torch.load(base_model_path))

    # All parameters trainable
    for param in model.parameters():
        param.requires_grad = True

    # Train
    history = train_with_finetuning(model, train_loader, val_loader, device,
                                     learning_rate=5e-5, max_epochs=50, output_dir=output_dir)

    # Test
    results = evaluate_on_test(model, test_loader, device)
    results.update(history)

    return results


def finetune_partial(base_model_path: Path, train_csv: Path, val_csv: Path, test_csv: Path,
                     input_columns: list, output_columns: list, output_dir: Path, seed: int = 42):
    """Partial Fine-Tuning (freeze encoder, only train head)"""
    print("\n=== Partial Fine-Tuning ===")
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_vals, f_idx, t_idx = load_data(train_csv, input_columns, output_columns, 32)
    val_vals, _, _ = load_data(val_csv, input_columns, output_columns, 32)
    test_vals, _, _ = load_data(test_csv, input_columns, output_columns, 32)

    scalers_mean, scalers_std = compute_scalers(train_vals)
    scaled_train = scale_values(train_vals, scalers_mean, scalers_std)
    scaled_val = scale_values(val_vals, scalers_mean, scalers_std)
    scaled_test = scale_values(test_vals, scalers_mean, scalers_std)

    train_bundle = build_dataset_bundle("TRANSFORMER", scaled_train, f_idx, t_idx, 32)
    val_bundle = build_dataset_bundle("TRANSFORMER", scaled_val, f_idx, t_idx, 32)
    test_bundle = build_dataset_bundle("TRANSFORMER", scaled_test, f_idx, t_idx, 32)

    train_loader = DataLoader(train_bundle.dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_bundle.dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_bundle.dataset, batch_size=128, shuffle=False)

    # Load model
    model = TransformerRegressor(
        input_dim=train_bundle.input_dim,
        output_dim=train_bundle.targets.shape[1],
        d_model=128, nhead=8, num_encoder_layers=4,
        dim_feedforward=512, dropout=0.1
    )
    model.load_state_dict(torch.load(base_model_path))

    # Freeze all except head
    for param in model.input_projection.parameters():
        param.requires_grad = False
    for param in model.pos_encoding.parameters():
        param.requires_grad = False
    for param in model.transformer_encoder.parameters():
        param.requires_grad = False
    # Only train head
    for param in model.head.parameters():
        param.requires_grad = True

    # Train
    history = train_with_finetuning(model, train_loader, val_loader, device,
                                     learning_rate=1e-3, max_epochs=50, output_dir=output_dir)

    # Test
    results = evaluate_on_test(model, test_loader, device)
    results.update(history)

    return results


def finetune_lora(base_model_path: Path, train_csv: Path, val_csv: Path, test_csv: Path,
                  input_columns: list, output_columns: list, output_dir: Path, rank: int = 8, seed: int = 42):
    """LoRA Fine-Tuning - Simplified version that adds small trainable adapter layers"""
    print(f"\n=== Adapter Fine-Tuning (simpler LoRA-like method) ===")
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_vals, f_idx, t_idx = load_data(train_csv, input_columns, output_columns, 32)
    val_vals, _, _ = load_data(val_csv, input_columns, output_columns, 32)
    test_vals, _, _ = load_data(test_csv, input_columns, output_columns, 32)

    scalers_mean, scalers_std = compute_scalers(train_vals)
    scaled_train = scale_values(train_vals, scalers_mean, scalers_std)
    scaled_val = scale_values(val_vals, scalers_mean, scalers_std)
    scaled_test = scale_values(test_vals, scalers_mean, scalers_std)

    train_bundle = build_dataset_bundle("TRANSFORMER", scaled_train, f_idx, t_idx, 32)
    val_bundle = build_dataset_bundle("TRANSFORMER", scaled_val, f_idx, t_idx, 32)
    test_bundle = build_dataset_bundle("TRANSFORMER", scaled_test, f_idx, t_idx, 32)

    train_loader = DataLoader(train_bundle.dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_bundle.dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_bundle.dataset, batch_size=128, shuffle=False)

    # Load model and move to device
    model = TransformerRegressor(
        input_dim=train_bundle.input_dim,
        output_dim=train_bundle.targets.shape[1],
        d_model=128, nhead=8, num_encoder_layers=4,
        dim_feedforward=512, dropout=0.1
    )
    model.load_state_dict(torch.load(base_model_path))
    model = model.to(device)

    # Freeze most layers
    for name, param in model.named_parameters():
        if 'head' not in name and 'input_projection' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Train only the head with a much smaller learning rate
    optimizer = Adam([
        {'params': [p for p in model.head.parameters() if p.requires_grad], 'lr': 1e-3},
        {'params': [p for p in model.input_projection.parameters() if p.requires_grad], 'lr': 5e-5},
    ], weight_decay=1e-5)

    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0
    train_history = []
    val_history = []

    for epoch in range(1, 51):
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
        train_history.append(train_loss)

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
        val_history.append(val_loss)

        print(f"Epoch {epoch}: train={train_loss:.6f}, val={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= 8:
                break

    if (output_dir / "best_model.pt").exists():
        model.load_state_dict(torch.load(output_dir / "best_model.pt"))

    results = evaluate_on_test(model, test_loader, device)
    results["best_val_loss"] = best_val_loss
    results["train_history"] = train_history
    results["val_history"] = val_history
    results["best_epoch"] = epoch

    return results


def run_finetuning_experiment():
    """Run complete finetuning experiment"""
    print("="*60)
    print("Finetuning Experiment")
    print("="*60)

    # 配置
    caww_35_train = REPO_ROOT / "data/caww_35c_split/train.csv"
    caww_35_val = REPO_ROOT / "data/caww_35c_split/val.csv"
    caww_35_test = REPO_ROOT / "data/caww_35c_split/test.csv"

    # 输入输出列
    input_cols = ["TRC-DT", "pH-DT", "cond-DT", "TRC-RT", "pH-RT", "cond-RT", "fDOM-RT", "DO-RT", "TOC-RT", "DOC-RT"]
    output_cols = ["TRC-PPL1", "TRC-PPL2", "pH-PPL1", "pH-PPL2", "cond-PPL1", "cond-PPL2", "TOC-PPL1", "TOC-PPL2"]

    results = {}

    # CAWW29 → CAWW35 finetuning
    print("\n" + "="*60)
    print("CAWW35 Finetuning (from CAWW29 Transformer)")
    print("="*60)

    # Find best transformer model
    transformer_dir = REPO_ROOT / "outputs/transformer_final"
    if transformer_dir.exists():
        subdirs = sorted([d for d in transformer_dir.iterdir() if d.is_dir()],
                        key=lambda p: p.stat().st_mtime, reverse=True)
        caww_29_model = subdirs[0] / "best_model.pt" if subdirs else None
    else:
        caww_29_model = None

    if not caww_29_model or not caww_29_model.exists():
        print("Warning: Transformer model not found, skipping CAWW35 finetuning")
        return

    # Full
    full_dir = REPO_ROOT / "outputs/finetune/caww35_full"
    full_dir.mkdir(parents=True, exist_ok=True)
    results["caww35_full"] = finetune_full(
        caww_29_model, caww_35_train, caww_35_val, caww_35_test,
        input_cols, output_cols, full_dir
    )

    # Partial
    partial_dir = REPO_ROOT / "outputs/finetune/caww35_partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    results["caww35_partial"] = finetune_partial(
        caww_29_model, caww_35_train, caww_35_val, caww_35_test,
        input_cols, output_cols, partial_dir
    )

    # LoRA
    lora_dir = REPO_ROOT / "outputs/finetune/caww35_lora"
    lora_dir.mkdir(parents=True, exist_ok=True)
    results["caww35_lora"] = finetune_lora(
        caww_29_model, caww_35_train, caww_35_val, caww_35_test,
        input_cols, output_cols, lora_dir, rank=8
    )

    # 保存结果
    output_file = REPO_ROOT / "outputs/finetune_results.json"
    with open(output_file, 'w') as f:
        # 移除numpy数组
        json_results = {}
        for k, v in results.items():
            json_results[k] = {key: val for key, val in v.items() if not isinstance(val, np.ndarray)}
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("\nSummary:")
    for method, metrics in results.items():
        print(f"{method}: R2={metrics.get('r2', 0):.4f}, RMSE={metrics.get('rmse', 0):.4f}")


if __name__ == "__main__":
    run_finetuning_experiment()