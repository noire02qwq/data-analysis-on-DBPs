#!/usr/bin/env python3
"""
Finetuning Transformer with LoRA and three methods:
1. Full Fine-Tuning - Update all model parameters
2. Partial Fine-Tuning - Freeze encoder, only train head
3. LoRA Fine-Tuning - Low-Rank Adaptation
"""

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
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
    """Low-Rank Adaptation layer for Transformer."""
    def __init__(self, d_model: int, rank: int = 8):
        super().__init__()
        self.rank = rank
        self.lora_a = nn.Linear(d_model, rank, bias=False)
        self.lora_b = nn.Linear(rank, d_model, bias=False)
        self.scaling = 1.0
        # Initialize LoRA params
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        return self.lora_b(self.lora_a(x)) * self.scaling


class LoRAWrapper(nn.Module):
    """Wrapper for Transformer with LoRA adaptation."""
    def __init__(self, base_model: nn.Module, rank: int = 8):
        super().__init__()
        self.base_model = base_model
        self.rank = rank

        # Add LoRA layers after each attention layer
        self.lora_layers = nn.ModuleList()
        for _ in range(len(base_model.transformer_encoder.layers)):
            self.lora_layers.append(LoRALayer(base_model.d_model, rank))

    def forward(self, x):
        # Get base model output
        x = self.base_model.input_projection(x)
        x = self.base_model.pos_encoding(x)

        for i, layer in enumerate(self.base_model.transformer_encoder.layers):
            # Original forward
            x = layer(x)
            # Add LoRA adaptation
            lora_out = self.lora_layers[i](x)
            x = x + lora_out * 0.1  # Scale LoRA contribution

        x = self.base_model.norm(x)
        x = x[:, -1, :]
        return self.base_model.head(x)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(train_csv, val_csv, test_csv, input_columns, output_columns):
    """Load and prepare data."""
    # Determine timestamp column
    train_path = Path(train_csv)
    df_temp = pl.read_csv(train_path, encoding="utf-8-sig", n_rows=1)
    ts_col = "Date, Time" if "Date, Time" in df_temp.columns else None

    df_train = load_time_series(train_path, ts_col) if ts_col else pl.read_csv(train_path)
    df_val = load_time_series(Path(val_csv), ts_col) if ts_col else pl.read_csv(val_csv)
    df_test = load_time_series(Path(test_csv), ts_col) if ts_col else pl.read_csv(test_csv)

    columns = df_train.columns

    feature_indices, target_indices = get_feature_and_target_indices(
        columns, input_columns, output_columns
    )

    values_train = df_train.to_numpy().astype(np.float32)
    values_val = df_val.to_numpy().astype(np.float32)
    values_test = df_test.to_numpy().astype(np.float32)

    scalers_mean, scalers_std = compute_scalers(values_train)

    scaled_train = scale_values(values_train, scalers_mean, scalers_std)
    scaled_val = scale_values(values_val, scalers_mean, scalers_std)
    scaled_test = scale_values(values_test, scalers_mean, scalers_std)

    return scaled_train, scaled_val, scaled_test, feature_indices, target_indices, scalers_mean, scalers_std


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    max_epochs: int = 50,
    patience: int = 8,
) -> Dict[str, Any]:
    """Train model with given config."""
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    train_history = []
    val_history = []

    for epoch in range(1, max_epochs + 1):
        # Training
        model.train()
        total_loss = 0
        n_samples = 0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

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
        total_loss = 0
        n_samples = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * features.size(0)
                n_samples += features.size(0)

        val_loss = total_loss / n_samples
        val_history.append(val_loss)

        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), "/tmp/best_model.pt")
        else:
            patience_counter += 1
            if patience > 0 and patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    model.load_state_dict(torch.load("/tmp/best_model.pt"))

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_history": train_history,
        "val_history": val_history,
    }


def test_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """Evaluate model on test set."""
    model.eval()
    criterion = nn.MSELoss()

    total_loss = 0
    n_samples = 0

    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            targets = targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * features.size(0)
            n_samples += features.size(0)

    return total_loss / n_samples


def finetune_full(model_path: str, output_path: str, data_config: Dict):
    """Full Fine-Tuning: Update all parameters."""
    print("\n=== Full Fine-Tuning ===")
    # Load pretrained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ... implementation
    pass


def finetune_partial(model_path: str, output_path: str, data_config: Dict):
    """Partial Fine-Tuning: Freeze encoder, only train head."""
    print("\n=== Partial Fine-Tuning ===")
    # Load model and freeze encoder
    pass


def finetune_lora(model_path: str, output_path: str, data_config: Dict):
    """LoRA Fine-Tuning: Add LoRA layers and train only those."""
    print("\n=== LoRA Fine-Tuning ===")
    # Load model, add LoRA layers
    pass


def main():
    parser = argparse.ArgumentParser(description="Finetune Transformer with LoRA")
    parser.add_argument("--base-model", required=True, help="Path to base Transformer model")
    parser.add_argument("--method", choices=["full", "partial", "lora"], required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    args = parser.parse_args()

    print(f"Finetuning method: {args.method}")
    print(f"Base model: {args.base_model}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()