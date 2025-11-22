#!/usr/bin/env python3
"""
Train rate-based regression models with OTHER (non-TRC) output targets.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import LSTMRegressor, MLPRegressor, RNNRegressor  # noqa: E402
from scripts.regression_utils import (  # noqa: E402
    DatasetBundle,
    SplitBoundaries,
    build_dataset_bundle,
    compute_scalers,
    compute_split_boundaries,
    get_feature_and_target_indices,
    load_time_series,
    scale_values,
    subset_indices,
    compute_rate_targets,
    rate_to_absolute,
)

# OTHER (non-TRC) target columns
OTHER_TARGET_COLUMNS: List[str] = [
    "TOC-PPL1",
    "TOC-PPL2",
    "DOC-PPL1",
    "DOC-PPL2",
    "pH-PPL1",
    "pH-PPL2",
]

SUPPORTED_MODELS = {"MLP", "MLP_WITH_HISTORY", "LSTM", "RNN", "GRU", "TRANSFORMER"}
PATIENCE_EPOCHS = 50
PATIENCE_BASE_EPOCH = 30
PATIENCE_START_EPOCH = 31
MIN_EPOCHS = 80
FORCE_STOP_EPOCH = 100
BEST_START_RATIO = 3.0


def safe_filename(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


@dataclass(frozen=True)
class ConfigBundle:
    model_type: str
    model_name: str
    model_params: Dict[str, Any]
    training_params: Dict[str, Any]
    data_params: Dict[str, Any]
    config_path: Path


def parse_config(path: Path) -> ConfigBundle:
    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError("YAML root must be a mapping.")

    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    if not isinstance(model_cfg, dict):
        raise ValueError("model section must be a mapping.")
    if not isinstance(training_cfg, dict):
        raise ValueError("training section must be a mapping.")
    if not isinstance(data_cfg, dict):
        raise ValueError("data section must be a mapping.")

    model_type = str(model_cfg["type"]).upper()
    model_name = str(model_cfg["name"])
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model type '{model_type}'.")

    model_params = dict(model_cfg)
    model_params.pop("type", None)
    model_params.pop("name", None)

    training_params = {
        "max_epochs": int(training_cfg.get("max_epochs", 400)),
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

    return ConfigBundle(
        model_type=model_type,
        model_name=model_name,
        model_params=model_params,
        training_params=training_params,
        data_params=data_params,
        config_path=path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train rate-based regression models with OTHER (non-TRC) targets.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_output_dirs(model_name: str, config_path: Path) -> Tuple[Path, Path]:
    # 修改输出目录命名规范为"模型-输出目标-回归目标"格式
    # 修复输出目录路径，使其与贝叶斯优化器期望的路径一致
    # 从model_name中提取trial信息
    if "/trial_" in model_name:
        # 如果是贝叶斯优化的trial运行，使用model_name中的trial信息
        parts = model_name.split("/")
        base_model_name = parts[0]
        trial_info = parts[1]
        # 修复目录命名逻辑，避免重复添加后缀
        if base_model_name.endswith("-other-rate"):
            # 如果base_model_name已经包含了-other-rate后缀，直接使用
            output_dir = Path("scripts") / "outputs" / base_model_name / trial_info
        else:
            # 否则添加-other-rate后缀
            output_dir = Path("scripts") / "outputs" / f"{base_model_name}-other-rate" / trial_info
    else:
        # 如果是普通运行，使用原始命名方式
        output_dir = Path("scripts") / "outputs" / f"{model_name}-other-rate"
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / config_path.name
    if config_path.resolve() != destination.resolve():
        shutil.copy(config_path, destination)
    return output_dir, checkpoints_dir


def get_other_feature_and_target_indices(columns: List[str]) -> Tuple[List[int], List[int]]:
    column_to_idx = {col: idx for idx, col in enumerate(columns)}
    target_indices: List[int] = []
    for col in OTHER_TARGET_COLUMNS:
        if col not in column_to_idx:
            missing = set(OTHER_TARGET_COLUMNS) - set(column_to_idx)
            raise ValueError(f"Missing target columns: {missing}")
        target_indices.append(column_to_idx[col])

    # Exclude all PPL columns from features
    from scripts.regression_utils import is_ppl_column
    feature_indices = [idx for idx, name in enumerate(columns) if not is_ppl_column(name)]
    if not feature_indices:
        raise ValueError("No non-PPL columns remain for features.")
    return feature_indices, target_indices


def build_torch_model(
    model_type: str,
    input_dim: int,
    output_dim: int,
    sequence_length: int | None,
    model_params: Dict[str, Any],
) -> nn.Module:
    if model_type == "MLP":
        return MLPRegressor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=model_params["hidden_layers"],
            dropout=float(model_params.get("dropout", 0.0)),
        )
    if model_type == "MLP_WITH_HISTORY":
        return MLPRegressor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=model_params["hidden_layers"],
            dropout=float(model_params.get("dropout", 0.0)),
        )
    if model_type == "LSTM":
        if sequence_length is None:
            raise ValueError("Sequence length is required for LSTM inputs.")
        return LSTMRegressor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=int(model_params.get("units", 192)),
            num_layers=int(model_params.get("num_layers", 2)),
            dropout=float(model_params.get("dropout", 0.0)),
            fc_dim=model_params.get("fc_dim"),
        )
    if model_type == "RNN":
        if sequence_length is None:
            raise ValueError("Sequence length is required for RNN inputs.")
        return RNNRegressor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=int(model_params.get("units", 160)),
            num_layers=int(model_params.get("num_layers", 2)),
            dropout=float(model_params.get("dropout", 0.0)),
            fc_dim=model_params.get("fc_dim"),
        )
    if model_type == "GRU":
        if sequence_length is None:
            raise ValueError("Sequence length is required for GRU inputs.")
        from models.gru_regressor import GRURegressor
        return GRURegressor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=int(model_params.get("units", 192)),
            num_layers=int(model_params.get("num_layers", 2)),
            dropout=float(model_params.get("dropout", 0.0)),
            fc_dim=model_params.get("fc_dim"),
        )
    if model_type == "TRANSFORMER":
        if sequence_length is None:
            raise ValueError("Sequence length is required for TRANSFORMER inputs.")
        from models.transformer_regressor import TransformerRegressor
        return TransformerRegressor(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=int(model_params.get("d_model", 128)),
            nhead=int(model_params.get("nhead", 8)),
            num_encoder_layers=int(model_params.get("num_encoder_layers", 4)),
            dim_feedforward=int(model_params.get("dim_feedforward", 512)),
            dropout=float(model_params.get("dropout", 0.1)),
            fc_dim=model_params.get("fc_dim"),
        )
    raise ValueError(f"Unsupported torch model_type '{model_type}'.")


def save_torch_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, val_loss: float) -> None:
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
    }, path)


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
        outputs = model(features)
        loss = criterion(outputs, targets)
        if training:
            loss.backward()
            optimizer.step()
        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


def plot_training_curve(
    epochs: List[int],
    train_losses: List[float],
    val_losses: List[float],
    output_path: Path,
    train_label: str = "Train Loss",
    val_label: str = "Val Loss",
) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_losses, label=train_label)
    plt.plot(epochs, val_losses, label=val_label)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_loss_history(path: Path, epochs: List[int], train_losses: List[float], val_losses: List[float]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("epoch,train_loss,val_loss\n")
        for epoch, train_loss, val_loss in zip(epochs, train_losses, val_losses):
            fh.write(f"{epoch},{train_loss:.6f},{val_loss:.6f}\n")


def train_with_torch(
    cfg: ConfigBundle,
    dataset_bundle: DatasetBundle,
    splits: Dict[str, np.ndarray],
    training_params: Dict[str, Any],
    output_dir: Path,
    checkpoints_dir: Path,
    device: torch.device,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> Dict[str, Any]:
    if dataset_bundle.base_targets is None:
        raise ValueError("Rate-based training requires base_targets in the dataset bundle.")

    dataset = dataset_bundle.dataset
    train_subset = Subset(dataset, splits["train"])
    val_subset = Subset(dataset, splits["val"])
    test_subset = Subset(dataset, splits["test"])
    
    train_loader = DataLoader(train_subset, batch_size=training_params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=training_params["batch_size"], shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=training_params["batch_size"], shuffle=False)

    # Get base values for validation and testing
    val_base = dataset_bundle.base_targets[np.array(val_subset.indices, dtype=int)]
    test_base = dataset_bundle.base_targets[np.array(test_subset.indices, dtype=int)]

    model = build_torch_model(
        cfg.model_type,
        dataset_bundle.input_dim,
        dataset_bundle.targets.shape[1],
        dataset_bundle.sequence_length,
        cfg.model_params,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_params["learning_rate"],
        weight_decay=training_params["weight_decay"],
    )
    train_criterion = nn.MSELoss()  # Rate-based loss
    val_criterion = nn.MSELoss()    # Rate-based loss
    best_val_loss = math.inf
    best_epoch = None
    best_model_saved = False
    train_history: List[float] = []
    val_history: List[float] = []
    epochs_axis: List[int] = []
    best_train_loss = math.inf
    initial_train_loss = None
    train_threshold = None
    best_tracking = False
    patience_active = False
    patience_best_train = None
    patience_best_val = None
    patience_no_improve_epochs = 0
    forced_stop_due_to_threshold = False

    max_epochs = training_params["max_epochs"]
    best_model_path = output_dir / "best_model.pt"
    last_model_path = output_dir / "last_model.pt"

    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(model, train_loader, train_criterion, optimizer, device)
        val_loss = run_epoch(model, val_loader, val_criterion, None, device)

        epochs_axis.append(epoch)
        train_history.append(train_loss)
        val_history.append(val_loss)

        if initial_train_loss is None:
            initial_train_loss = train_loss
            train_threshold = train_loss / BEST_START_RATIO if train_loss > 0 else train_loss

        if not best_tracking and train_threshold is not None and train_loss <= train_threshold + 1e-12:
            best_tracking = True
            best_val_loss = val_loss
            best_epoch = epoch
            save_torch_checkpoint(best_model_path, model, optimizer, epoch, val_loss)
            best_model_saved = True
        elif best_tracking and val_loss < best_val_loss - 1e-9:
            best_val_loss = val_loss
            best_epoch = epoch
            save_torch_checkpoint(best_model_path, model, optimizer, epoch, val_loss)
            best_model_saved = True

        if not patience_active and (best_tracking or epoch >= PATIENCE_START_EPOCH):
            patience_active = True
            patience_best_val = val_loss
            patience_no_improve_epochs = 0

        if patience_active:
            if patience_best_val is None or val_loss < patience_best_val - 1e-9:
                patience_best_val = val_loss
                patience_no_improve_epochs = 0
            else:
                patience_no_improve_epochs += 1
                if patience_no_improve_epochs >= PATIENCE_EPOCHS and epoch >= MIN_EPOCHS:
                    break

        if not best_tracking and epoch >= FORCE_STOP_EPOCH:
            forced_stop_due_to_threshold = True
            break

        log_msg = f"[{cfg.model_name}][Epoch {epoch:03d}] train_mse={train_loss:.6f} val_mse={val_loss:.6f}"
        print(log_msg)

    save_torch_checkpoint(last_model_path, model, optimizer, epochs_axis[-1], val_history[-1])
    if not best_model_saved:
        shutil.copy(last_model_path, best_model_path)
        best_val_loss = val_history[-1]
        best_epoch = epochs_axis[-1]

    # Load best model for testing
    best_state = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_state["model_state"])
    
    # Evaluate on test set (rate-based)
    test_loss = run_epoch(model, test_loader, train_criterion, None, device)
    
    # Convert predictions to absolute values for final evaluation
    model.eval()
    
    # Get predictions on validation set
    val_preds_scaled = []
    val_targets_scaled = []
    with torch.no_grad():
        for features, targets in val_loader:
            features = features.to(device)
            outputs = model(features)
            val_preds_scaled.append(outputs.cpu().numpy())
            val_targets_scaled.append(targets.numpy())
    
    val_preds_scaled_np = np.concatenate(val_preds_scaled, axis=0)
    val_targets_scaled_np = np.concatenate(val_targets_scaled, axis=0)
    
    # Convert to absolute values
    val_preds_abs = rate_to_absolute(val_preds_scaled_np, val_base)
    val_targets_abs = rate_to_absolute(val_targets_scaled_np, val_base)
    
    # Calculate MSE on absolute values
    val_abs_mse = np.mean((val_preds_abs - val_targets_abs) ** 2)
    
    # Get predictions on test set
    test_preds_scaled = []
    test_targets_scaled = []
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            outputs = model(features)
            test_preds_scaled.append(outputs.cpu().numpy())
            test_targets_scaled.append(targets.numpy())
    
    test_preds_scaled_np = np.concatenate(test_preds_scaled, axis=0)
    test_targets_scaled_np = np.concatenate(test_targets_scaled, axis=0)
    
    # Convert to absolute values
    test_preds_abs = rate_to_absolute(test_preds_scaled_np, test_base)
    test_targets_abs = rate_to_absolute(test_targets_scaled_np, test_base)
    
    # Calculate MSE on absolute values
    test_abs_mse = np.mean((test_preds_abs - test_targets_abs) ** 2)

    return {
        "train_losses": train_history,
        "val_losses": val_history,  # Rate-based validation losses
        "epochs": epochs_axis,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,  # Rate-based best validation loss
        "best_val_abs_mse": float(val_abs_mse),  # Absolute value MSE for best model on validation
        "test_loss": test_loss,  # Rate-based test loss
        "test_abs_mse": float(test_abs_mse),  # Absolute value MSE for best model on test
        "tracker_state": {
            "initial_train_loss": initial_train_loss,
            "train_threshold": train_threshold,
            "best_tracking": best_tracking,
            "patience_active": patience_active,
            "patience_best_train": patience_best_train,
            "patience_best_val": patience_best_val,
            "patience_no_improve_epochs": patience_no_improve_epochs,
            "best_train_loss": best_train_loss,
            "forced_stop_due_to_threshold": forced_stop_due_to_threshold,
        },
        "best_model_path": str(best_model_path),
        "last_model_path": str(last_model_path),
        "model_format": "torch",
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = parse_config(config_path)
    output_dir, checkpoints_dir = configure_output_dirs(cfg.model_name, config_path)
    set_seed(cfg.training_params["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.model_type != "XGBOOST" else "cpu")

    df = load_time_series(cfg.data_params["input_csv"], cfg.data_params["timestamp_column"])
    columns = list(df.columns)
    
    # Compute rate targets
    rate_targets, base_targets, base_columns = compute_rate_targets(df, OTHER_TARGET_COLUMNS)
    
    feature_indices, target_indices = get_other_feature_and_target_indices(columns)
    feature_columns = [columns[idx] for idx in feature_indices]

    boundaries = compute_split_boundaries(len(df))
    values = df.to_numpy(dtype=np.float32)
    
    # Scale features using standard approach
    scalers_mean, scalers_std = compute_scalers(values, boundaries.train_end)
    scaled_values = scale_values(values, scalers_mean, scalers_std)
    
    # Scale rate targets
    target_values = rate_targets
    target_train_slice = target_values[:boundaries.train_end]
    target_mean = target_train_slice.mean(axis=0)
    target_std = target_train_slice.std(axis=0)
    target_std[target_std == 0] = 1.0
    scaled_targets = (target_values - target_mean) / target_std
    
    # Build dataset with base targets
    dataset_bundle = build_dataset_bundle(
        cfg.model_type,
        scaled_values,
        feature_indices,
        target_indices,
        cfg.model_params["history_length"],
        base_targets=scaled_targets,  # Use scaled rate targets as the targets
    )
    splits = subset_indices(dataset_bundle.valid_indices, boundaries)

    training_result = train_with_torch(
        cfg,
        dataset_bundle,
        splits,
        cfg.training_params,
        output_dir,
        checkpoints_dir,
        device,
        target_mean,
        target_std,
    )
    plot_training_curve(
        training_result["epochs"],
        training_result["train_losses"],
        training_result["val_losses"],
        output_dir / "training_curve.png",
        train_label="Train MSE",
        val_label="Val MSE",
    )
    save_loss_history(
        output_dir / "loss_history.csv",
        training_result["epochs"],
        training_result["train_losses"],
        training_result["val_losses"],
    )
    
    # Save scalers for both features and targets
    np.savez_compressed(
        output_dir / "scalers.npz",
        feature_mean=scalers_mean,
        feature_std=scalers_std,
        target_mean=target_mean,
        target_std=target_std,
    )

    dataset_sizes = {name: int(idx.size) for name, idx in splits.items()}
    training_history = {
        "epochs": training_result["epochs"],
        "train_loss": training_result["train_losses"],
        "val_loss": training_result["val_losses"],
        "best_epoch": training_result["best_epoch"],
        "best_val_loss": training_result["best_val_loss"],
        "best_val_abs_mse": training_result["best_val_abs_mse"],
        "test_loss": training_result["test_loss"],
        "test_abs_mse": training_result["test_abs_mse"],
        "tracker": training_result["tracker_state"],
    }

    metadata = {
        "model_name": cfg.model_name,
        "model_type": cfg.model_type,
        "model_format": training_result["model_format"],
        "model_params": cfg.model_params,
        "training_params": cfg.training_params,
        "data_csv": str(cfg.data_params["input_csv"]),
        "timestamp_column": cfg.data_params["timestamp_column"],
        "columns": columns,
        "feature_columns": feature_columns,
        "target_columns": OTHER_TARGET_COLUMNS,
        "target_mode": "rate",
        "target_base_columns": base_columns,
        "split_boundaries": {
            "train_end": boundaries.train_end,
            "val_end": boundaries.val_end,
            "test_end": boundaries.test_end,
        },
        "dataset_sizes": dataset_sizes,
        "input_dim": dataset_bundle.input_dim,
        "sequence_length": dataset_bundle.sequence_length,
        "training_history": training_history,
        "model_files": {
            "best": training_result["best_model_path"],
            "last": training_result["last_model_path"],
        },
        "config_path": str(cfg.config_path),
    }

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)

    print(f"Training artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()