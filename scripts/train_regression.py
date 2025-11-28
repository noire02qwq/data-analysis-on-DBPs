#!/usr/bin/env python3
"""
Train regression models (MLP/MLP_WITH_HISTORY/LSTM/RNN/XGBOOST) per 回归实验1109.md.
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
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import LSTMRegressor, MLPRegressor, RNNRegressor, XGBoostRegressor  # noqa: E402
from scripts.regression_utils import (  # noqa: E402
    DatasetBundle,
    SplitBoundaries,
    TARGET_COLUMNS,
    build_dataset_bundle,
    compute_scalers,
    compute_split_boundaries,
    get_feature_and_target_indices,
    load_time_series,
    scale_values,
    subset_indices,
)

SUPPORTED_MODELS = {"MLP", "MLP_WITH_HISTORY", "LSTM", "RNN", "XGBOOST", "LIGHTGBM", "CATBOOST", "GRU"}
PATIENCE_EPOCHS = 50
PATIENCE_BASE_EPOCH = 30
PATIENCE_START_EPOCH = 31
MIN_EPOCHS = 80
FORCE_STOP_EPOCH = 100
BEST_START_RATIO = 3.0


def safe_filename(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


@dataclass
class ConfigBundle:
    model_type: str
    model_name: str
    model_params: Dict[str, Any]
    training_params: Dict[str, Any]
    data_params: Dict[str, Any]
    config_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train regression models for DBPS data.")
    parser.add_argument(
        "--config",
        default="models/configs/mlp_config.yaml",
        help="Path to a YAML config file.",
    )
    return parser.parse_args()


def parse_config(path: Path) -> ConfigBundle:
    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError("YAML root must be a mapping.")

    model_cfg = config.get("model")
    training_cfg = config.get("training")
    data_cfg = config.get("data")
    if not all(isinstance(section, dict) for section in (model_cfg, training_cfg, data_cfg)):
        raise ValueError("Config must contain 'model', 'training', and 'data' sections.")

    model_type = str(model_cfg.get("type", "")).strip().upper()
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model type '{model_type}'. Expected {sorted(SUPPORTED_MODELS)}.")

    model_name = str(model_cfg.get("name") or f"{model_type.lower()}_model").strip()
    history_length = int(model_cfg.get("history_length", 1))
    model_params: Dict[str, Any] = {"history_length": history_length}

    if model_type in {"MLP", "MLP_WITH_HISTORY"}:
        hidden_layers_raw = model_cfg.get("hidden_layers", [512, 256, 128])
        if isinstance(hidden_layers_raw, str):
            hidden_layers = [int(chunk.strip()) for chunk in hidden_layers_raw.split(",") if chunk.strip()]
        else:
            hidden_layers = [int(val) for val in hidden_layers_raw]
        model_params["hidden_layers"] = hidden_layers
        model_params["dropout"] = float(model_cfg.get("dropout", 0.0))
    elif model_type in {"LSTM", "RNN"}:
        model_params["units"] = int(model_cfg.get("units", 192))
        model_params["num_layers"] = int(model_cfg.get("num_layers", 2))
        model_params["dropout"] = float(model_cfg.get("dropout", 0.0))
        model_params["fc_dim"] = model_cfg.get("fc_dim")
    elif model_type == "XGBOOST":
        model_params["max_depth"] = int(model_cfg.get("max_depth", 8))
        model_params["learning_rate"] = float(model_cfg.get("learning_rate", 0.05))
        model_params["subsample"] = float(model_cfg.get("subsample", 0.9))
        model_params["colsample_bytree"] = float(model_cfg.get("colsample_bytree", 0.8))
        model_params["gamma"] = float(model_cfg.get("gamma", 0.0))
        model_params["reg_lambda"] = float(model_cfg.get("reg_lambda", 1.0))
        model_params["min_child_weight"] = float(model_cfg.get("min_child_weight", 1.0))
    elif model_type == "LIGHTGBM":
        model_params["num_leaves"] = int(model_cfg.get("num_leaves", 255))
        model_params["max_depth"] = int(model_cfg.get("max_depth", -1))
        model_params["learning_rate"] = float(model_cfg.get("learning_rate", 0.05))
        model_params["subsample"] = float(model_cfg.get("subsample", 0.9))
        model_params["colsample_bytree"] = float(model_cfg.get("colsample_bytree", 0.8))
        model_params["min_child_samples"] = int(model_cfg.get("min_child_samples", 40))
        model_params["reg_alpha"] = float(model_cfg.get("reg_alpha", 0.0))
        model_params["reg_lambda"] = float(model_cfg.get("reg_lambda", 1.0))
        model_params["bagging_freq"] = int(model_cfg.get("bagging_freq", 1))
    else:  # CATBOOST
        model_params["depth"] = int(model_cfg.get("depth", 8))
        model_params["learning_rate"] = float(model_cfg.get("learning_rate", 0.05))
        model_params["l2_leaf_reg"] = float(model_cfg.get("l2_leaf_reg", 3.0))
        model_params["subsample"] = float(model_cfg.get("subsample", 0.8))
        model_params["random_strength"] = float(model_cfg.get("random_strength", 1.0))
        model_params["bagging_temperature"] = float(model_cfg.get("bagging_temperature", 1.0))

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_output_dirs(model_name: str, config_path: Path) -> Tuple[Path, Path]:
    output_dir = Path("scripts") / "outputs" / model_name
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / config_path.name
    if config_path.resolve() != destination.resolve():
        shutil.copy(config_path, destination)
    return output_dir, checkpoints_dir


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
    raise ValueError(f"Unsupported torch model_type '{model_type}'.")


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


def save_torch_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
        },
        path,
    )


def plot_training_curve(
    epochs: List[int],
    train_losses: List[float],
    val_losses: List[float],
    destination: Path,
    train_label: str = "Train Loss",
    val_label: str = "Val Loss",
) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_losses, label=train_label)
    plt.plot(epochs, val_losses, label=val_label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(destination)
    plt.close()


def save_loss_history(path: Path, epochs: List[int], train_losses: List[float], val_losses: List[float]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("epoch,train_loss,val_loss\n")
        for epoch, tr, vl in zip(epochs, train_losses, val_losses):
            fh.write(f"{epoch},{tr},{vl}\n")


def save_scalers(path: Path, mean: np.ndarray, std: np.ndarray) -> None:
    np.savez(path, mean=mean, std=std)


def plot_per_target_curves(per_target_history: Dict[str, Dict[str, List[float]]], output_dir: Path) -> None:
    for target, history in per_target_history.items():
        plot_training_curve(
            history["epochs"],
            history["train_losses"],
            history["val_losses"],
            output_dir / f"training_curve_{safe_filename(target)}.png",
            train_label="Train MSE",
            val_label="Val MSE",
        )


def save_loss_history_multi(per_target_history: Dict[str, Dict[str, List[float]]], output_dir: Path) -> None:
    for target, history in per_target_history.items():
        path = output_dir / f"loss_history_{safe_filename(target)}.csv"
        save_loss_history(path, history["epochs"], history["train_losses"], history["val_losses"])


def train_single_target_xgb(
    target_name: str,
    params: Dict[str, Any],
    training_params: Dict[str, Any],
    train_features: np.ndarray,
    val_features: np.ndarray,
    test_features: np.ndarray,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    output_dir: Path,
) -> Dict[str, Any]:
    dtrain = xgb.DMatrix(train_features, label=train_labels)
    dval = xgb.DMatrix(val_features, label=val_labels)
    dtest = xgb.DMatrix(test_features, label=test_labels)

    booster: xgb.Booster | None = None
    train_history: List[float] = []
    val_history: List[float] = []
    epochs_axis: List[int] = []
    initial_train_loss = None
    train_threshold = None
    best_tracking = False
    patience_active = False
    patience_best_val = None
    patience_no_improve_epochs = 0
    best_val_loss = math.inf
    best_epoch = None
    forced_stop_due_to_threshold = False
    best_model_saved = False

    best_dir = output_dir / "best_model"
    last_dir = output_dir / "last_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    last_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = best_dir / f"{target_name}.json"
    last_model_path = last_dir / f"{target_name}.json"
    max_epochs = training_params["max_epochs"]

    for epoch in range(1, max_epochs + 1):
        booster = xgb.train(
            params,
            dtrain=dtrain,
            num_boost_round=1,
            xgb_model=booster,
            verbose_eval=False,
        )
        train_pred = booster.predict(dtrain)
        val_pred = booster.predict(dval)
        train_loss = float(np.mean((train_pred - train_labels) ** 2))
        val_loss = float(np.mean((val_pred - val_labels) ** 2))

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
            booster.save_model(best_model_path)
            best_model_saved = True
        elif best_tracking and val_loss < best_val_loss - 1e-9:
            best_val_loss = val_loss
            best_epoch = epoch
            booster.save_model(best_model_path)
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

        log_msg = f"[{target_name}][Epoch {epoch:03d}] train_mse={train_loss:.6f} val_mse={val_loss:.6f}"
        print(log_msg)

    if booster is None:
        raise RuntimeError("XGBoost booster failed to train.")

    booster.save_model(last_model_path)
    if not best_model_saved:
        shutil.copy(last_model_path, best_model_path)
        best_val_loss = val_history[-1]
        best_epoch = epochs_axis[-1]

    best_booster = xgb.Booster()
    best_booster.load_model(best_model_path)
    test_pred = best_booster.predict(dtest)
    test_loss = float(np.mean((test_pred - test_labels) ** 2))

    return {
        "epochs": epochs_axis,
        "train_losses": train_history,
        "val_losses": val_history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "tracking_enabled": best_tracking,
        "quarter_reached": train_threshold is not None,
        "patience_best_train": initial_train_loss,
        "patience_best_val": patience_best_val,
        "forced_stop_due_to_threshold": forced_stop_due_to_threshold,
    }


def train_with_torch(
    cfg: ConfigBundle,
    dataset_bundle: DatasetBundle,
    splits: Dict[str, np.ndarray],
    training_params: Dict[str, Any],
    output_dir: Path,
    checkpoints_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    dataset = dataset_bundle.dataset
    train_loader = DataLoader(Subset(dataset, splits["train"]), batch_size=training_params["batch_size"], shuffle=True)
    val_loader = DataLoader(Subset(dataset, splits["val"]), batch_size=training_params["batch_size"], shuffle=False)
    test_loader = DataLoader(Subset(dataset, splits["test"]), batch_size=training_params["batch_size"], shuffle=False)

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
    train_criterion = nn.MSELoss()
    val_criterion = nn.MSELoss()
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

        if train_loss < best_train_loss - 1e-9:
            best_train_loss = train_loss

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
            patience_best_train = train_loss
            patience_best_val = val_loss
            patience_no_improve_epochs = 0

        if patience_active:
            improved = False
            if patience_best_train is None or train_loss < patience_best_train - 1e-9:
                patience_best_train = train_loss
                improved = True
            if patience_best_val is None or val_loss < patience_best_val - 1e-9:
                patience_best_val = val_loss
                improved = True
            if improved:
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

    best_state = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_state["model_state"])
    test_loss = run_epoch(model, test_loader, train_criterion, None, device)

    return {
        "train_losses": train_history,
        "val_losses": val_history,
        "epochs": epochs_axis,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
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


def train_with_xgboost(
    cfg: ConfigBundle,
    bundle: DatasetBundle,
    splits: Dict[str, np.ndarray],
    training_params: Dict[str, Any],
    output_dir: Path,
    checkpoints_dir: Path,
) -> Dict[str, Any]:
    params = {
        "objective": "reg:squarederror",
        "max_depth": cfg.model_params["max_depth"],
        "eta": cfg.model_params["learning_rate"],
        "subsample": cfg.model_params["subsample"],
        "colsample_bytree": cfg.model_params["colsample_bytree"],
        "gamma": cfg.model_params["gamma"],
        "lambda": cfg.model_params["reg_lambda"],
        "min_child_weight": cfg.model_params["min_child_weight"],
        "verbosity": 0,
    }

    train_features = bundle.features[splits["train"]]
    val_features = bundle.features[splits["val"]]
    test_features = bundle.features[splits["test"]]
    train_targets = bundle.targets[splits["train"]]
    val_targets = bundle.targets[splits["val"]]
    test_targets = bundle.targets[splits["test"]]

    per_target_history: Dict[str, Dict[str, Any]] = {}
    best_dir = output_dir / "best_model"
    last_dir = output_dir / "last_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    last_dir.mkdir(parents=True, exist_ok=True)

    for idx, target_name in enumerate(TARGET_COLUMNS):
        history = train_single_target_xgb(
            target_name=target_name,
            params=params,
            training_params=training_params,
            train_features=train_features,
            val_features=val_features,
            test_features=test_features,
            train_labels=train_targets[:, idx],
            val_labels=val_targets[:, idx],
            test_labels=test_targets[:, idx],
            output_dir=output_dir,
        )
        per_target_history[target_name] = history

    avg_best_val_loss = float(np.mean([hist["best_val_loss"] for hist in per_target_history.values()]))
    avg_test_loss = float(np.mean([hist["test_loss"] for hist in per_target_history.values()]))
    avg_best_epoch = float(np.mean([hist["best_epoch"] or 0 for hist in per_target_history.values()]))
    primary_target = "TRC-PPL1" if "TRC-PPL1" in per_target_history else next(iter(per_target_history))
    primary_best_val = float(per_target_history[primary_target]["best_val_loss"])

    return {
        "per_target_history": per_target_history,
        "avg_best_val_loss": avg_best_val_loss,
        "avg_test_loss": avg_test_loss,
        "avg_best_epoch": avg_best_epoch,
        "primary_target": primary_target,
        "primary_best_val_loss": primary_best_val,
        "best_model_path": str(best_dir),
        "last_model_path": str(last_dir),
        "model_format": "xgboost",
    }


def train_with_lightgbm(
    cfg: ConfigBundle,
    bundle: DatasetBundle,
    splits: Dict[str, np.ndarray],
    training_params: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    train_features = bundle.features[splits["train"]]
    val_features = bundle.features[splits["val"]]
    test_features = bundle.features[splits["test"]]
    train_targets = bundle.targets[splits["train"]]
    val_targets = bundle.targets[splits["val"]]
    test_targets = bundle.targets[splits["test"]]

    per_target_history: Dict[str, Dict[str, Any]] = {}
    best_dir = output_dir / "best_model"
    last_dir = output_dir / "last_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    last_dir.mkdir(parents=True, exist_ok=True)

    for idx, target_name in enumerate(TARGET_COLUMNS):
        history, booster = train_single_target_lightgbm(
            target_name,
            cfg.model_params,
            training_params,
            train_features,
            val_features,
            test_features,
            train_targets[:, idx],
            val_targets[:, idx],
            test_targets[:, idx],
        )
        per_target_history[target_name] = history
        model_path = best_dir / f"{target_name}.txt"
        booster.save_model(model_path)
        shutil.copy(model_path, last_dir / f"{target_name}.txt")

    avg_best_val_loss = float(np.mean([hist["best_val_loss"] for hist in per_target_history.values()]))
    avg_test_loss = float(np.mean([hist["test_loss"] for hist in per_target_history.values()]))
    avg_best_epoch = float(np.mean([hist.get("best_epoch", 0) for hist in per_target_history.values()]))
    primary_target = "TRC-PPL1" if "TRC-PPL1" in per_target_history else next(iter(per_target_history))
    primary_best_val_loss = float(per_target_history[primary_target]["best_val_loss"])

    return {
        "per_target_history": per_target_history,
        "avg_best_val_loss": avg_best_val_loss,
        "avg_test_loss": avg_test_loss,
        "avg_best_epoch": avg_best_epoch,
        "primary_target": primary_target,
        "primary_best_val_loss": primary_best_val_loss,
        "best_model_path": str(best_dir),
        "last_model_path": str(last_dir),
        "model_format": "lightgbm",
    }


def train_single_target_catboost(
    target_name: str,
    params: Dict[str, Any],
    training_params: Dict[str, Any],
    train_features: np.ndarray,
    val_features: np.ndarray,
    test_features: np.ndarray,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
) -> Tuple[Dict[str, Any], CatBoostRegressor]:
    model = CatBoostRegressor(
        iterations=int(training_params.get("max_epochs", 400)),
        depth=int(params.get("depth", 8)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        l2_leaf_reg=float(params.get("l2_leaf_reg", 3.0)),
        subsample=float(params.get("subsample", 0.8)),
        random_strength=float(params.get("random_strength", 1.0)),
        bagging_temperature=float(params.get("bagging_temperature", 1.0)),
        loss_function="RMSE",
        eval_metric="RMSE",
        od_type="Iter",
        od_wait=PATIENCE_EPOCHS,
        random_seed=int(training_params.get("seed", 42)),
        verbose=False,
    )
    train_pool = Pool(train_features, train_labels)
    val_pool = Pool(val_features, val_labels)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=False)
    evals_result = model.get_evals_result()
    train_rmse = evals_result.get("learn", {}).get("RMSE", [])
    val_rmse = evals_result.get("validation", {}).get("RMSE", [])
    train_history = [float(val) ** 2 for val in train_rmse]
    val_history = [float(val) ** 2 for val in val_rmse]
    if not train_history:
        train_history = [float(np.mean((model.predict(train_features) - train_labels) ** 2))]
    if not val_history:
        val_history = [float(np.mean((model.predict(val_features) - val_labels) ** 2))]
    best_iter = model.get_best_iteration()
    if best_iter is None or best_iter < 0:
        best_iter = len(val_history) - 1
    best_epoch = best_iter + 1
    best_val_loss = float(min(val_history[: best_iter + 1] or val_history))
    test_pred = model.predict(test_features)
    test_loss = float(np.mean((test_pred - test_labels) ** 2))
    history = {
        "epochs": list(range(1, len(val_history) + 1)),
        "train_losses": train_history,
        "val_losses": val_history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
    }
    return history, model


def train_with_catboost(
    cfg: ConfigBundle,
    bundle: DatasetBundle,
    splits: Dict[str, np.ndarray],
    training_params: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    train_features = bundle.features[splits["train"]]
    val_features = bundle.features[splits["val"]]
    test_features = bundle.features[splits["test"]]
    train_targets = bundle.targets[splits["train"]]
    val_targets = bundle.targets[splits["val"]]
    test_targets = bundle.targets[splits["test"]]

    per_target_history: Dict[str, Dict[str, Any]] = {}
    best_dir = output_dir / "best_model"
    last_dir = output_dir / "last_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    last_dir.mkdir(parents=True, exist_ok=True)

    for idx, target_name in enumerate(TARGET_COLUMNS):
        history, model = train_single_target_catboost(
            target_name,
            cfg.model_params,
            training_params,
            train_features,
            val_features,
            test_features,
            train_targets[:, idx],
            val_targets[:, idx],
            test_targets[:, idx],
        )
        per_target_history[target_name] = history
        best_path = best_dir / f"{target_name}.cbm"
        model.save_model(best_path)
        model.save_model(last_dir / f"{target_name}.cbm")

    avg_best_val_loss = float(np.mean([hist["best_val_loss"] for hist in per_target_history.values()]))
    avg_test_loss = float(np.mean([hist["test_loss"] for hist in per_target_history.values()]))
    avg_best_epoch = float(np.mean([hist.get("best_epoch", 0) for hist in per_target_history.values()]))
    primary_target = "TRC-PPL1" if "TRC-PPL1" in per_target_history else next(iter(per_target_history))
    primary_best_val_loss = float(per_target_history[primary_target]["best_val_loss"])

    return {
        "per_target_history": per_target_history,
        "avg_best_val_loss": avg_best_val_loss,
        "avg_test_loss": avg_test_loss,
        "avg_best_epoch": avg_best_epoch,
        "primary_target": primary_target,
        "primary_best_val_loss": primary_best_val_loss,
        "best_model_path": str(best_dir),
        "last_model_path": str(last_dir),
        "model_format": "catboost",
    }


def train_single_target_lightgbm(
    target_name: str,
    params: Dict[str, Any],
    training_params: Dict[str, Any],
    train_features: np.ndarray,
    val_features: np.ndarray,
    test_features: np.ndarray,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
) -> Tuple[Dict[str, Any], lgb.Booster]:
    evals_result: Dict[str, Dict[str, List[float]]] = {}
    model = lgb.LGBMRegressor(
        boosting_type=params.get("boosting_type", "gbdt"),
        num_leaves=int(params.get("num_leaves", 255)),
        max_depth=int(params.get("max_depth", -1)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        subsample=float(params.get("subsample", 0.9)),
        colsample_bytree=float(params.get("colsample_bytree", 0.8)),
        min_child_samples=int(params.get("min_child_samples", 40)),
        reg_alpha=float(params.get("reg_alpha", 0.0)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
        bagging_freq=int(params.get("bagging_freq", 1)),
        n_estimators=int(training_params.get("max_epochs", 400)),
        objective="regression",
        random_state=int(training_params.get("seed", 42)),
    )
    callbacks = [lgb.record_evaluation(evals_result)]
    callbacks.append(lgb.early_stopping(PATIENCE_EPOCHS, verbose=False))
    model.fit(
        train_features,
        train_labels,
        eval_set=[(train_features, train_labels), (val_features, val_labels)],
        eval_names=["train", "val"],
        eval_metric="l2",
        callbacks=callbacks,
    )
    train_history = [float(x) for x in evals_result.get("train", {}).get("l2", [])]
    val_history = [float(x) for x in evals_result.get("val", {}).get("l2", [])]
    if not train_history:
        train_history = [float(np.mean((model.predict(train_features) - train_labels) ** 2))]
    if not val_history:
        val_history = [float(np.mean((model.predict(val_features) - val_labels) ** 2))]
    best_iteration = model.best_iteration_
    if best_iteration is None or best_iteration <= 0:
        best_iteration = len(val_history)
    best_val_loss = float(min(val_history[:best_iteration] or val_history))
    test_pred = model.predict(test_features, num_iteration=model.best_iteration_)
    test_loss = float(np.mean((test_pred - test_labels) ** 2))
    history = {
        "epochs": list(range(1, len(val_history) + 1)),
        "train_losses": train_history,
        "val_losses": val_history,
        "best_epoch": best_iteration,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
    }
    booster = model.booster_
    if booster is None:
        raise RuntimeError("LightGBM booster unavailable after training.")
    return history, booster


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
    feature_indices, target_indices = get_feature_and_target_indices(columns)
    feature_columns = [columns[idx] for idx in feature_indices]

    boundaries = compute_split_boundaries(len(df))
    values = df.to_numpy(dtype=np.float32)
    scalers_mean, scalers_std = compute_scalers(values, boundaries.train_end)
    scaled_values = scale_values(values, scalers_mean, scalers_std)

    dataset_bundle = build_dataset_bundle(
        cfg.model_type,
        scaled_values,
        feature_indices,
        target_indices,
        cfg.model_params["history_length"],
    )
    splits = subset_indices(dataset_bundle.valid_indices, boundaries)

    if cfg.model_type == "XGBOOST":
        training_result = train_with_xgboost(
            cfg,
            dataset_bundle,
            splits,
            cfg.training_params,
            output_dir,
            checkpoints_dir,
        )
        plot_per_target_curves(training_result["per_target_history"], output_dir)
        save_loss_history_multi(training_result["per_target_history"], output_dir)
    elif cfg.model_type == "LIGHTGBM":
        training_result = train_with_lightgbm(
            cfg,
            dataset_bundle,
            splits,
            cfg.training_params,
            output_dir,
        )
        plot_per_target_curves(training_result["per_target_history"], output_dir)
        save_loss_history_multi(training_result["per_target_history"], output_dir)
    elif cfg.model_type == "CATBOOST":
        training_result = train_with_catboost(
            cfg,
            dataset_bundle,
            splits,
            cfg.training_params,
            output_dir,
        )
        plot_per_target_curves(training_result["per_target_history"], output_dir)
        save_loss_history_multi(training_result["per_target_history"], output_dir)
    else:
        training_result = train_with_torch(
            cfg,
            dataset_bundle,
            splits,
            cfg.training_params,
            output_dir,
            checkpoints_dir,
            device,
        )
        plot_training_curve(
            training_result["epochs"],
            training_result["train_losses"],
            training_result["val_losses"],
            output_dir / "training_curve.png",
            train_label="Train MSE",
            val_label="Val MAE",
        )
        save_loss_history(
            output_dir / "loss_history.csv",
            training_result["epochs"],
            training_result["train_losses"],
            training_result["val_losses"],
        )
    save_scalers(output_dir / "scalers.npz", scalers_mean, scalers_std)

    dataset_sizes = {name: int(idx.size) for name, idx in splits.items()}
    if cfg.model_type in {"XGBOOST", "LIGHTGBM", "CATBOOST"}:
        training_history = {
            "per_target": training_result["per_target_history"],
            "avg_best_val_loss": training_result["avg_best_val_loss"],
            "avg_test_loss": training_result["avg_test_loss"],
            "avg_best_epoch": training_result["avg_best_epoch"],
            "primary_target": training_result["primary_target"],
            "primary_best_val_loss": training_result["primary_best_val_loss"],
        }
    else:
        training_history = {
            "epochs": training_result["epochs"],
            "train_loss": training_result["train_losses"],
            "val_loss": training_result["val_losses"],
            "best_epoch": training_result["best_epoch"],
            "best_val_loss": training_result["best_val_loss"],
            "test_loss": training_result["test_loss"],
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
        "target_columns": TARGET_COLUMNS,
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
