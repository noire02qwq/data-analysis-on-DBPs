#!/usr/bin/env python3
"""
Unified training script for DBPs regression models.
Supports MLP, RNN, LSTM, GRU, TRANSFORMER, XGBoost, LightGBM, and CatBoost.
Reads input/output columns from a TOML configuration file.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import shutil
import sys
import tomli
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import LSTMRegressor, MLPRegressor, RNNRegressor, XGBoostRegressor
from scripts.utils import (
    DatasetBundle,
    build_dataset_bundle,
    compute_scalers,
    get_feature_and_target_indices,
    load_time_series,
    scale_values,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SUPPORTED_MODELS = {"MLP", "LSTM", "RNN", "GRU", "TRANSFORMER", "XGBOOST", "LIGHTGBM", "CATBOOST"}


@dataclass
class ConfigBundle:
    model_type: str
    model_name: str
    model_params: Dict[str, Any]
    training_params: Dict[str, Any]
    data_params: Dict[str, Any]
    config_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified training script.")
    parser.add_argument("--config", required=True, help="Path to TOML config file.")
    return parser.parse_args()


def parse_config(path: Path) -> ConfigBundle:
    with path.open("rb") as fh:
        config = tomli.load(fh)

    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    model_type = str(model_cfg.get("type", "")).strip().upper()
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model type '{model_type}'. Expected {sorted(SUPPORTED_MODELS)}.")

    model_name = str(model_cfg.get("name") or f"{model_type.lower()}_model").strip()
    history_length = int(model_cfg.get("history_length", 1))
    model_params: Dict[str, Any] = {"history_length": history_length}

    if model_type == "MLP":
        model_params["hidden_layers"] = model_cfg.get("hidden_layers", [512, 256, 128])
        model_params["dropout"] = float(model_cfg.get("dropout", 0.0))
    elif model_type in {"LSTM", "RNN", "GRU"}:
        model_params["units"] = int(model_cfg.get("units", 192))
        model_params["num_layers"] = int(model_cfg.get("num_layers", 2))
        model_params["dropout"] = float(model_cfg.get("dropout", 0.0))
        model_params["fc_dim"] = model_cfg.get("fc_dim")
    elif model_type == "TRANSFORMER":
        model_params["d_model"] = int(model_cfg.get("d_model", 128))
        model_params["nhead"] = int(model_cfg.get("nhead", 8))
        model_params["num_encoder_layers"] = int(model_cfg.get("num_encoder_layers", 4))
        model_params["dim_feedforward"] = int(model_cfg.get("dim_feedforward", 512))
        model_params["dropout"] = float(model_cfg.get("dropout", 0.1))
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
    elif model_type == "CATBOOST":
        model_params["depth"] = int(model_cfg.get("depth", 8))
        model_params["learning_rate"] = float(model_cfg.get("learning_rate", 0.05))
        model_params["l2_leaf_reg"] = float(model_cfg.get("l2_leaf_reg", 3.0))
        model_params["subsample"] = float(model_cfg.get("subsample", 0.8))
        model_params["random_strength"] = float(model_cfg.get("random_strength", 1.0))
        model_params["bagging_temperature"] = float(model_cfg.get("bagging_temperature", 1.0))

    training_params = {
        "max_epochs": int(training_cfg.get("max_epochs", 100)),
        "batch_size": int(training_cfg.get("batch_size", 256)),
        "learning_rate": float(training_cfg.get("learning_rate", 1e-3)),
        "weight_decay": float(training_cfg.get("weight_decay", 0.0)),
        "patience": int(training_cfg.get("patience", 50)),
        "seed": int(training_cfg.get("seed", 42)),
    }

    data_params = {
        "train_csv": Path(str(data_cfg["train_csv"])),
        "val_csv": Path(str(data_cfg["val_csv"])),
        "test_csv": Path(str(data_cfg["test_csv"])),
        "input_columns": list(data_cfg.get("input_columns", [])),
        "output_columns": list(data_cfg.get("output_columns", [])),
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


def build_torch_model(
    model_type: str,
    input_dim: int,
    output_dim: int,
    model_params: Dict[str, Any],
) -> nn.Module:
    if model_type == "MLP":
        return MLPRegressor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=model_params["hidden_layers"],
            dropout=model_params["dropout"],
        )
    if model_type == "LSTM":
        return LSTMRegressor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=model_params["units"],
            num_layers=model_params["num_layers"],
            dropout=model_params["dropout"],
            fc_dim=model_params.get("fc_dim"),
        )
    if model_type == "RNN":
        return RNNRegressor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=model_params["units"],
            num_layers=model_params["num_layers"],
            dropout=model_params["dropout"],
            fc_dim=model_params.get("fc_dim"),
        )
    if model_type == "GRU":
        from models.gru_regressor import GRURegressor
        return GRURegressor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=model_params["units"],
            num_layers=model_params["num_layers"],
            dropout=model_params["dropout"],
            fc_dim=model_params.get("fc_dim"),
        )
    if model_type == "TRANSFORMER":
        from models.transformer_regressor import TransformerRegressor
        return TransformerRegressor(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=model_params["d_model"],
            nhead=model_params["nhead"],
            num_encoder_layers=model_params["num_encoder_layers"],
            dim_feedforward=model_params["dim_feedforward"],
            dropout=model_params["dropout"],
            fc_dim=model_params.get("fc_dim"),
        )
    raise ValueError(f"Unsupported torch model: {model_type}")


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


def save_loss_history(path: Path, epochs: List[int], train_losses: List[float], val_losses: List[float]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("epoch,train_loss,val_loss\n")
        for epoch, tr, vl in zip(epochs, train_losses, val_losses):
            fh.write(f"{epoch},{tr},{vl}\n")


def plot_training_curve(
    epochs: List[int],
    train_losses: List[float],
    val_losses: List[float],
    destination: Path,
) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(destination)
    plt.close()


def export_torch_onnx(model: nn.Module, input_dim: int, sequence_length: int | None, path: Path, device: torch.device) -> None:
    model.eval()
    if sequence_length is not None:
        dummy_input = torch.randn(1, sequence_length, input_dim, device=device)
        dynamic_axes = {'input': {0: 'batch_size', 1: 'sequence_length'}, 'output': {0: 'batch_size'}}
    else:
        dummy_input = torch.randn(1, input_dim, device=device)
        dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        
    torch.onnx.export(
        model,
        dummy_input,
        str(path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
    )


def train_with_torch(
    cfg: ConfigBundle,
    train_bundle: DatasetBundle,
    val_bundle: DatasetBundle,
    test_bundle: DatasetBundle,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    
    train_loader = DataLoader(train_bundle.dataset, batch_size=cfg.training_params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_bundle.dataset, batch_size=cfg.training_params["batch_size"], shuffle=False)
    test_loader = DataLoader(test_bundle.dataset, batch_size=cfg.training_params["batch_size"], shuffle=False)

    model = build_torch_model(
        cfg.model_type,
        train_bundle.input_dim,
        train_bundle.targets.shape[1],
        cfg.model_params,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training_params["learning_rate"],
        weight_decay=cfg.training_params["weight_decay"],
    )
    criterion = nn.MSELoss()
    
    max_epochs = cfg.training_params["max_epochs"]
    patience = cfg.training_params["patience"]
    
    best_val_loss = math.inf
    best_epoch = 0
    patience_counter = 0

    train_history: List[float] = []
    val_history: List[float] = []
    epochs_axis: List[int] = []

    best_model_path = output_dir / "best_model.pt"
    last_model_path = output_dir / "last_model.pt"
    onnx_model_path = output_dir / "best_model.onnx"

    logging.info(f"Starting {cfg.model_type} training for {max_epochs} epochs on {device}...")

    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = run_epoch(model, val_loader, criterion, None, device)

        epochs_axis.append(epoch)
        train_history.append(train_loss)
        val_history.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Epoch {epoch:03d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f} (New Best)")
        else:
            patience_counter += 1
            logging.info(f"Epoch {epoch:03d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
            if patience > 0 and patience_counter >= patience:
                logging.info(f"Early stopping triggered at epoch {epoch}")
                break

    torch.save(model.state_dict(), last_model_path)

    # Evaluate on test set with best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss = run_epoch(model, test_loader, criterion, None, device)

    # Export ONNX (optional, skip if onnxscript not available)
    try:
        export_torch_onnx(model, train_bundle.input_dim, train_bundle.sequence_length, onnx_model_path, device)
    except ImportError:
        logging.warning("ONNX export skipped: onnxscript not installed")

    return {
        "model_format": "torch",
        "epochs": epochs_axis,
        "train_losses": train_history,
        "val_losses": val_history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
    }

# XGBoost, LightGBM, CatBoost require separate models per output target
def train_with_xgboost(
    cfg: ConfigBundle,
    train_bundle: DatasetBundle,
    val_bundle: DatasetBundle,
    test_bundle: DatasetBundle,
    output_dir: Path,
) -> Dict[str, Any]:
    output_columns = cfg.data_params["output_columns"]
    max_epochs = cfg.training_params["max_epochs"]
    patience = cfg.training_params["patience"]
    
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
        "seed": cfg.training_params["seed"],
    }

    per_target_history = {}

    for idx, target in enumerate(output_columns):
        logging.info(f"Training XGBoost for target {target}...")
        dtrain = xgb.DMatrix(train_bundle.features, label=train_bundle.targets[:, idx])
        dval = xgb.DMatrix(val_bundle.features, label=val_bundle.targets[:, idx])
        dtest = xgb.DMatrix(test_bundle.features, label=test_bundle.targets[:, idx])
        
        evals_result = {}
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=max_epochs,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=patience if patience > 0 else None,
            evals_result=evals_result,
            verbose_eval=False,
        )
        
        train_history = evals_result["train"]["rmse"]
        val_history = evals_result["val"]["rmse"]
        # Convert RMSE to MSE
        train_history = [v**2 for v in train_history]
        val_history = [v**2 for v in val_history]
        
        best_epoch = booster.best_iteration + 1
        best_val_loss = val_history[best_epoch - 1]
        
        test_pred = booster.predict(dtest)
        test_loss = float(np.mean((test_pred - test_bundle.targets[:, idx]) ** 2))
        
        booster.save_model(output_dir / f"best_model_{target}.xgb")
        
        per_target_history[target] = {
            "epochs": list(range(1, len(val_history) + 1)),
            "train_losses": train_history,
            "val_losses": val_history,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "test_loss": test_loss,
        }
    
    avg_best_val_loss = np.mean([h["best_val_loss"] for h in per_target_history.values()])
    avg_test_loss = np.mean([h["test_loss"] for h in per_target_history.values()])

    return {
        "model_format": "xgboost",
        "per_target_history": per_target_history,
        "best_val_loss": float(avg_best_val_loss),
        "test_loss": float(avg_test_loss),
    }


def train_with_lightgbm(
    cfg: ConfigBundle,
    train_bundle: DatasetBundle,
    val_bundle: DatasetBundle,
    test_bundle: DatasetBundle,
    output_dir: Path,
) -> Dict[str, Any]:
    output_columns = cfg.data_params["output_columns"]
    max_epochs = cfg.training_params["max_epochs"]
    patience = cfg.training_params["patience"]
    
    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "l2",
        "num_leaves": cfg.model_params["num_leaves"],
        "max_depth": cfg.model_params["max_depth"],
        "learning_rate": cfg.model_params["learning_rate"],
        "subsample": cfg.model_params["subsample"],
        "colsample_bytree": cfg.model_params["colsample_bytree"],
        "min_child_samples": cfg.model_params["min_child_samples"],
        "reg_alpha": cfg.model_params["reg_alpha"],
        "reg_lambda": cfg.model_params["reg_lambda"],
        "bagging_freq": cfg.model_params["bagging_freq"],
        "verbose": -1,
        "seed": cfg.training_params["seed"],
    }

    per_target_history = {}

    for idx, target in enumerate(output_columns):
        logging.info(f"Training LightGBM for target {target}...")
        lgb_train = lgb.Dataset(train_bundle.features, train_bundle.targets[:, idx])
        lgb_val = lgb.Dataset(val_bundle.features, val_bundle.targets[:, idx], reference=lgb_train)
        
        evals_result = {}
        callbacks = [lgb.record_evaluation(evals_result)]
        if patience > 0:
            callbacks.append(lgb.early_stopping(patience, verbose=False))
            
        booster = lgb.train(
            params,
            lgb_train,
            num_boost_round=max_epochs,
            valid_sets=[lgb_train, lgb_val],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )
        
        train_history = evals_result["train"]["l2"]
        val_history = evals_result["val"]["l2"]
        
        best_epoch = booster.best_iteration
        best_val_loss = val_history[best_epoch - 1]
        
        test_pred = booster.predict(test_bundle.features)
        test_loss = float(np.mean((test_pred - test_bundle.targets[:, idx]) ** 2))
        
        booster.save_model(output_dir / f"best_model_{target}.lgb")
        
        per_target_history[target] = {
            "epochs": list(range(1, len(val_history) + 1)),
            "train_losses": train_history,
            "val_losses": val_history,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "test_loss": test_loss,
        }
    
    avg_best_val_loss = np.mean([h["best_val_loss"] for h in per_target_history.values()])
    avg_test_loss = np.mean([h["test_loss"] for h in per_target_history.values()])

    return {
        "model_format": "lightgbm",
        "per_target_history": per_target_history,
        "best_val_loss": float(avg_best_val_loss),
        "test_loss": float(avg_test_loss),
    }


def train_with_catboost(
    cfg: ConfigBundle,
    train_bundle: DatasetBundle,
    val_bundle: DatasetBundle,
    test_bundle: DatasetBundle,
    output_dir: Path,
) -> Dict[str, Any]:
    output_columns = cfg.data_params["output_columns"]
    max_epochs = cfg.training_params["max_epochs"]
    patience = cfg.training_params["patience"]
    
    per_target_history = {}

    for idx, target in enumerate(output_columns):
        logging.info(f"Training CatBoost for target {target}...")
        train_pool = Pool(train_bundle.features, train_bundle.targets[:, idx])
        val_pool = Pool(val_bundle.features, val_bundle.targets[:, idx])
        
        model = CatBoostRegressor(
            iterations=max_epochs,
            depth=cfg.model_params["depth"],
            learning_rate=cfg.model_params["learning_rate"],
            l2_leaf_reg=cfg.model_params["l2_leaf_reg"],
            subsample=cfg.model_params["subsample"],
            random_strength=cfg.model_params["random_strength"],
            bagging_temperature=cfg.model_params["bagging_temperature"],
            loss_function="RMSE",
            eval_metric="RMSE",
            early_stopping_rounds=patience if patience > 0 else None,
            random_seed=cfg.training_params["seed"],
            verbose=False,
        )
        
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        
        evals_result = model.get_evals_result()
        train_rmse = evals_result["learn"]["RMSE"]
        val_rmse = evals_result["validation"]["RMSE"]
        
        train_history = [v**2 for v in train_rmse]
        val_history = [v**2 for v in val_rmse]
        
        best_epoch = model.get_best_iteration() + 1
        best_val_loss = val_history[best_epoch - 1]
        
        test_pred = model.predict(test_bundle.features)
        test_loss = float(np.mean((test_pred - test_bundle.targets[:, idx]) ** 2))
        
        model.save_model(output_dir / f"best_model_{target}.cbm")
        
        per_target_history[target] = {
            "epochs": list(range(1, len(val_history) + 1)),
            "train_losses": train_history,
            "val_losses": val_history,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "test_loss": test_loss,
        }
    
    avg_best_val_loss = np.mean([h["best_val_loss"] for h in per_target_history.values()])
    avg_test_loss = np.mean([h["test_loss"] for h in per_target_history.values()])

    return {
        "model_format": "catboost",
        "per_target_history": per_target_history,
        "best_val_loss": float(avg_best_val_loss),
        "test_loss": float(avg_test_loss),
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = parse_config(config_path)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = REPO_ROOT / "outputs" / cfg.model_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    shutil.copy(config_path, output_dir / "config.toml")
    
    set_seed(cfg.training_params["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Loading pre-split datasets...")
    # Check if "Date, Time" column exists
    train_csv_path = REPO_ROOT / cfg.data_params["train_csv"]
    df_temp = pl.read_csv(train_csv_path, encoding="utf-8-sig", n_rows=1)
    ts_col = "Date, Time" if "Date, Time" in df_temp.columns else None

    df_train = load_time_series(REPO_ROOT / cfg.data_params["train_csv"], ts_col)
    df_val = load_time_series(REPO_ROOT / cfg.data_params["val_csv"], ts_col)
    df_test = load_time_series(REPO_ROOT / cfg.data_params["test_csv"], ts_col)

    columns = df_train.columns
    input_cols = cfg.data_params["input_columns"]
    output_cols = cfg.data_params["output_columns"]
    
    feature_indices, target_indices = get_feature_and_target_indices(columns, input_cols, output_cols)

    values_train = df_train.to_numpy().astype(np.float32)
    values_val = df_val.to_numpy().astype(np.float32)
    values_test = df_test.to_numpy().astype(np.float32)

    scalers_mean, scalers_std = compute_scalers(values_train)
    np.savez_compressed(output_dir / "scalers.npz", mean=scalers_mean, std=scalers_std)

    scaled_train = scale_values(values_train, scalers_mean, scalers_std)
    scaled_val = scale_values(values_val, scalers_mean, scalers_std)
    scaled_test = scale_values(values_test, scalers_mean, scalers_std)

    history_length = cfg.model_params["history_length"]

    train_bundle = build_dataset_bundle(cfg.model_type, scaled_train, feature_indices, target_indices, history_length)
    val_bundle = build_dataset_bundle(cfg.model_type, scaled_val, feature_indices, target_indices, history_length)
    test_bundle = build_dataset_bundle(cfg.model_type, scaled_test, feature_indices, target_indices, history_length)

    if cfg.model_type == "XGBOOST":
        result = train_with_xgboost(cfg, train_bundle, val_bundle, test_bundle, output_dir)
        for target, hist in result["per_target_history"].items():
            save_loss_history(output_dir / f"loss_history_{target}.csv", hist["epochs"], hist["train_losses"], hist["val_losses"])
            plot_training_curve(hist["epochs"], hist["train_losses"], hist["val_losses"], output_dir / f"training_curve_{target}.png")
    elif cfg.model_type == "LIGHTGBM":
        result = train_with_lightgbm(cfg, train_bundle, val_bundle, test_bundle, output_dir)
        for target, hist in result["per_target_history"].items():
            save_loss_history(output_dir / f"loss_history_{target}.csv", hist["epochs"], hist["train_losses"], hist["val_losses"])
            plot_training_curve(hist["epochs"], hist["train_losses"], hist["val_losses"], output_dir / f"training_curve_{target}.png")
    elif cfg.model_type == "CATBOOST":
        result = train_with_catboost(cfg, train_bundle, val_bundle, test_bundle, output_dir)
        for target, hist in result["per_target_history"].items():
            save_loss_history(output_dir / f"loss_history_{target}.csv", hist["epochs"], hist["train_losses"], hist["val_losses"])
            plot_training_curve(hist["epochs"], hist["train_losses"], hist["val_losses"], output_dir / f"training_curve_{target}.png")
    else:
        result = train_with_torch(cfg, train_bundle, val_bundle, test_bundle, output_dir, device)
        save_loss_history(output_dir / "loss_history.csv", result["epochs"], result["train_losses"], result["val_losses"])
        plot_training_curve(result["epochs"], result["train_losses"], result["val_losses"], output_dir / "training_curve.png")

    result_toml = {
        "model_name": cfg.model_name,
        "model_type": cfg.model_type,
        "model_format": result["model_format"],
        "input_columns": input_cols,
        "output_columns": output_cols,
        "eval": {
            "best_val_loss": result["best_val_loss"],
            "test_loss": result["test_loss"],
        }
    }
    
    if "best_epoch" in result:
        result_toml["eval"]["best_epoch"] = result["best_epoch"]

    with (output_dir / "result.toml").open("w", encoding="utf-8") as f:
        for k, v in result_toml.items():
            if isinstance(v, dict):
                f.write(f"\n[{k}]\n")
                for sub_k, sub_v in v.items():
                    f.write(f"{sub_k} = {sub_v}\n")
            elif isinstance(v, list):
                if not v:
                    f.write(f"{k} = []\n")
                elif isinstance(v[0], str):
                    formatted_list = ", ".join(f'"{item}"' for item in v)
                    f.write(f"{k} = [{formatted_list}]\n")
                else:
                    f.write(f"{k} = {v}\n")
            else:
                if isinstance(v, str):
                    f.write(f'{k} = "{v}"\n')
                else:
                    f.write(f"{k} = {v}\n")

    logging.info(f"Training completed successfully! Output saved to: {output_dir}")

if __name__ == "__main__":
    main()
