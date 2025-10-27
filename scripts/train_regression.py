#!/usr/bin/env python3
"""
Train regression models (MLP/LSTM) for the DBPS experiment according to the YAML config.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("PyYAML is required. Install it via 'pip install pyyaml'.") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import LSTMRegressor, MLPRegressor
from scripts.regression_utils import (
    LSTMSequenceDataset,
    MLPPastSequenceDataset,
    SplitBoundaries,
    compute_scalers,
    compute_split_boundaries,
    get_column_indices,
    load_time_series,
    scale_values,
    subset_indices,
    TARGET_COLUMNS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLP/LSTM regression models.")
    parser.add_argument(
        "--config",
        default="models/configs/mlp_config.yaml",
        help="Path to the YAML config describing the experiment.",
    )
    return parser.parse_args()


def parse_config(config_path: Path) -> Dict[str, object]:
    with config_path.open("r", encoding="utf-8") as fh:
        config_data = yaml.safe_load(fh)
    if not isinstance(config_data, dict):
        raise ValueError("Config root must be a mapping.")

    model_cfg = config_data.get("model")
    training_cfg = config_data.get("training")
    data_cfg = config_data.get("data")
    if not isinstance(model_cfg, dict) or not isinstance(training_cfg, dict) or not isinstance(data_cfg, dict):
        raise ValueError("Config must contain 'model', 'training', and 'data' sections.")

    model_type = str(model_cfg.get("type", "")).strip().upper()
    if model_type not in {"MLP", "LSTM"}:
        raise ValueError("Model type must be either 'MLP' or 'LSTM'.")
    model_name = str(model_cfg.get("name") or f"{model_type.lower()}_model").strip()
    history_length = int(model_cfg["history_length"])

    model_params: Dict[str, Any] = {"history_length": history_length}

    if model_type == "MLP":
        hidden_layers_raw = model_cfg.get("hidden_layers", [256, 128])
        if isinstance(hidden_layers_raw, str):
            hidden_layers = [int(layer.strip()) for layer in hidden_layers_raw.split(",") if layer.strip()]
        elif isinstance(hidden_layers_raw, (list, tuple)):
            hidden_layers = [int(val) for val in hidden_layers_raw]
        else:
            raise ValueError("hidden_layers must be a list or comma-separated string.")
        if not hidden_layers:
            raise ValueError("At least one hidden layer is required for the MLP.")
        model_params["hidden_layers"] = hidden_layers
        model_params["dropout"] = float(model_cfg.get("dropout", 0.0))
    else:
        model_params["hidden_size"] = int(model_cfg.get("hidden_size", 128))
        model_params["num_layers"] = int(model_cfg.get("num_layers", 2))
        model_params["dropout"] = float(model_cfg.get("dropout", 0.0))
        fc_dim = model_cfg.get("fc_dim")
        model_params["fc_dim"] = int(fc_dim) if fc_dim is not None else None

    training_params = {
        "epochs": int(training_cfg.get("epochs", 50)),
        "batch_size": int(training_cfg.get("batch_size", 256)),
        "learning_rate": float(training_cfg.get("learning_rate", 1e-3)),
        "weight_decay": float(training_cfg.get("weight_decay", 0.0)),
        "checkpoint_interval": int(training_cfg.get("checkpoint_interval", 10)),
        "seed": int(training_cfg.get("seed", 42)),
    }

    data_params = {
        "input_csv": Path(str(data_cfg["input_csv"])),
        "timestamp_column": str(data_cfg.get("timestamp_column", "Date, Time")),
    }

    return {
        "model_type": model_type,
        "model_name": model_name,
        "model_params": model_params,
        "training_params": training_params,
        "data_params": data_params,
        "config_path": str(config_path),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_datasets(
    model_type: str,
    model_params: Dict[str, object],
    scaled_values: np.ndarray,
    non_target_indices: List[int],
    target_indices: List[int],
) -> Tuple[torch.utils.data.Dataset, int, int]:
    non_target_data = scaled_values[:, non_target_indices]
    target_data = scaled_values[:, target_indices]

    if model_type == "MLP":
        dataset = MLPPastSequenceDataset(
            scaled_values,
            non_target_data,
            target_data,
            history_length=model_params["history_length"],
        )
        input_dim = dataset.input_dim
    else:
        dataset = LSTMSequenceDataset(
            non_target_data,
            target_data,
            history_length=model_params["history_length"],
        )
        input_dim = dataset.input_dim
    output_dim = target_data.shape[1]
    return dataset, input_dim, output_dim


def build_model(model_type: str, input_dim: int, output_dim: int, model_params: Dict[str, object]) -> nn.Module:
    if model_type == "MLP":
        return MLPRegressor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=model_params["hidden_layers"],
            dropout=float(model_params.get("dropout", 0.0)),
        )

    return LSTMRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_size=int(model_params.get("hidden_size", 128)),
        num_layers=int(model_params.get("num_layers", 2)),
        dropout=float(model_params.get("dropout", 0.0)),
        fc_dim=model_params.get("fc_dim"),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        if training:
            optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, targets)
        if training:
            loss.backward()
            optimizer.step()

        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("Dataset split is empty.")
    return total_loss / total_samples


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, val_loss: float) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
        },
        path,
    )


def plot_training_curve(epochs: List[int], train_losses: List[float], val_losses: List[float], destination: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(destination)
    plt.close()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = parse_config(config_path)

    model_type: str = cfg["model_type"]  # type: ignore[assignment]
    model_name: str = cfg["model_name"]  # type: ignore[assignment]
    model_params: Dict[str, object] = cfg["model_params"]  # type: ignore[assignment]
    training_params: Dict[str, object] = cfg["training_params"]  # type: ignore[assignment]
    data_params: Dict[str, object] = cfg["data_params"]  # type: ignore[assignment]

    output_dir = Path("scripts") / "outputs" / model_name
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(config_path, output_dir / config_path.name)

    set_seed(int(training_params["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_time_series(data_params["input_csv"], data_params["timestamp_column"])
    columns = list(df.columns)
    non_target_indices, target_indices = get_column_indices(columns)

    split_boundaries = compute_split_boundaries(len(df))
    column_values = df.to_numpy(dtype=np.float32)

    scalers_mean, scalers_std = compute_scalers(column_values, split_boundaries.train_end)
    scaled_values = scale_values(column_values, scalers_mean, scalers_std)

    dataset, input_dim, output_dim = build_datasets(
        model_type,
        model_params,
        scaled_values,
        non_target_indices,
        target_indices,
    )
    splits = subset_indices(dataset.valid_indices, split_boundaries)

    train_loader = DataLoader(
        Subset(dataset, splits["train"]),
        batch_size=int(training_params["batch_size"]),
        shuffle=True,
    )
    val_loader = DataLoader(
        Subset(dataset, splits["val"]),
        batch_size=int(training_params["batch_size"]),
        shuffle=False,
    )
    test_loader = DataLoader(
        Subset(dataset, splits["test"]),
        batch_size=int(training_params["batch_size"]),
        shuffle=False,
    )

    model = build_model(model_type, input_dim, output_dim, model_params).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_params["learning_rate"]),
        weight_decay=float(training_params["weight_decay"]),
    )
    criterion = nn.MSELoss()

    checkpoint_interval = max(1, int(training_params["checkpoint_interval"]))
    best_val_loss = math.inf
    best_epoch = None
    best_model_saved = False
    initial_train_loss = None
    train_threshold = None
    tracking_enabled = False

    train_history: List[float] = []
    val_history: List[float] = []
    epoch_axis: List[int] = []

    epochs = int(training_params["epochs"])
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = run_epoch(model, val_loader, criterion, None, device)

        if initial_train_loss is None:
            initial_train_loss = train_loss
            train_threshold = initial_train_loss / 4 if initial_train_loss > 0 else initial_train_loss

        epoch_axis.append(epoch)
        train_history.append(train_loss)
        val_history.append(val_loss)

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if not tracking_enabled and train_threshold is not None and train_loss <= train_threshold + 1e-12:
            tracking_enabled = True
            print(f"  Train loss crossed threshold ({train_threshold:.6f}); now tracking validation minima.")

        if tracking_enabled and val_loss < best_val_loss - 1e-9:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(output_dir / "best_model.pt", model, optimizer, epoch, val_loss)
            best_model_saved = True
            print("  Updated best model based on validation loss.")

        if epoch % checkpoint_interval == 0:
            ckpt_path = checkpoints_dir / f"epoch_{epoch:03d}.pt"
            save_checkpoint(ckpt_path, model, optimizer, epoch, val_loss)

    # Always save the final model state.
    save_checkpoint(output_dir / "last_model.pt", model, optimizer, epochs, val_history[-1])
    if not best_model_saved:
        shutil.copy(output_dir / "last_model.pt", output_dir / "best_model.pt")
        best_val_loss = val_history[-1]
        best_epoch = epochs
        print("Best model never triggered; using the last epoch weights as best_model.pt.")

    plot_training_curve(epoch_axis, train_history, val_history, output_dir / "training_curve.png")

    # Evaluate on test split with the best model.
    best_state = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_state["model_state"])
    test_loss = run_epoch(model, test_loader, criterion, None, device)
    print(f"Test loss: {test_loss:.6f}")

    # Persist metadata for reuse by the testing script.
    np.savez(output_dir / "scalers.npz", mean=scalers_mean, std=scalers_std)

    metadata = {
        "model_name": model_name,
        "model_type": model_type,
        "model_params": model_params,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "columns": columns,
        "non_target_columns": [columns[i] for i in non_target_indices],
        "target_columns": TARGET_COLUMNS,
        "data_csv": str(data_params["input_csv"]),
        "timestamp_column": data_params["timestamp_column"],
        "split_boundaries": {
            "train_end": split_boundaries.train_end,
            "val_end": split_boundaries.val_end,
            "test_end": split_boundaries.test_end,
        },
        "dataset_sizes": {name: int(idx.size) for name, idx in splits.items()},
        "training_history": {
            "epochs": epoch_axis,
            "train_loss": train_history,
            "val_loss": val_history,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "test_loss": test_loss,
            "train_threshold": train_threshold,
            "tracking_enabled": tracking_enabled,
            "initial_train_loss": initial_train_loss,
        },
        "config_path": cfg["config_path"],
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)

    history_csv = output_dir / "loss_history.csv"
    with history_csv.open("w", encoding="utf-8") as fh:
        fh.write("epoch,train_loss,val_loss\n")
        for epoch, tr, vl in zip(epoch_axis, train_history, val_history):
            fh.write(f"{epoch},{tr},{vl}\n")

    print(f"Training artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
