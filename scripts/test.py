#!/usr/bin/env python3
"""
Unified test script for evaluating trained regression models.
Supports PyTorch models (MLP, RNN, LSTM, GRU, Transformer) and GBDT models (XGBoost, LightGBM, CatBoost).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import LSTMRegressor, MLPRegressor, RNNRegressor
from scripts.utils import (
    DatasetBundle,
    build_dataset_bundle,
    compute_scalers,
    get_feature_and_target_indices,
    load_time_series,
    scale_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test trained regression models.")
    parser.add_argument("--model-dir", required=True, help="Directory containing trained model.")
    parser.add_argument("--test-csv", help="Test CSV file (optional, uses config from model).")
    parser.add_argument("--output-dir", help="Output directory for test results.")
    return parser.parse_args()


def load_result_toml(model_dir: Path) -> Dict[str, Any]:
    result_path = model_dir / "result.toml"
    if not result_path.exists():
        raise FileNotFoundError(f"result.toml not found in {model_dir}")
    import tomli
    with result_path.open("rb") as f:
        return tomli.load(f)


def load_scalers(model_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    scalers_path = model_dir / "scalers.npz"
    if not scalers_path.exists():
        raise FileNotFoundError(f"scalers.npz not found in {model_dir}")
    data = np.load(scalers_path)
    return data["mean"], data["std"]


def load_config(model_dir: Path) -> Dict[str, Any]:
    config_path = model_dir / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.toml not found in {model_dir}")
    import tomli
    with config_path.open("rb") as f:
        return tomli.load(f)


def plot_predictions(
    target_names: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path
) -> None:
    for idx, name in enumerate(target_names):
        plt.figure(figsize=(8, 4))
        plt.plot(y_true[:, idx], label="True", alpha=0.7)
        plt.plot(y_pred[:, idx], label="Predicted", alpha=0.7)
        plt.title(f"{name} - True vs Predicted")
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        safe_name = name.replace("/", "_").replace(" ", "_")
        plt.savefig(output_dir / f"{safe_name}_pred_vs_true.png")
        plt.close()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))

    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


def load_torch_model(model_dir: Path, model_type: str, input_dim: int, output_dim: int, device: torch.device) -> nn.Module:
    """Load a PyTorch model from checkpoint with config-based parameters."""
    # Load model params from config
    config_path = model_dir / "config.toml"
    model_params = {}
    if config_path.exists():
        import tomli
        with config_path.open("rb") as f:
            config = tomli.load(f)
        model_params = config.get("model", {})

    if model_type == "MLP":
        hidden_layers = model_params.get("hidden_layers", [512, 256, 128])
        model = MLPRegressor(input_dim=input_dim, output_dim=output_dim, hidden_layers=hidden_layers)
    elif model_type == "LSTM":
        hidden_size = model_params.get("units", 192)
        num_layers = model_params.get("num_layers", 2)
        dropout = model_params.get("dropout", 0.0)
        model = LSTMRegressor(input_dim=input_dim, output_dim=output_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    elif model_type == "RNN":
        hidden_size = model_params.get("units", 192)
        num_layers = model_params.get("num_layers", 2)
        dropout = model_params.get("dropout", 0.0)
        model = RNNRegressor(input_dim=input_dim, output_dim=output_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    elif model_type == "GRU":
        from models.gru_regressor import GRURegressor
        hidden_size = model_params.get("units", 192)
        num_layers = model_params.get("num_layers", 2)
        dropout = model_params.get("dropout", 0.0)
        model = GRURegressor(input_dim=input_dim, output_dim=output_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    elif model_type == "TRANSFORMER":
        from models.transformer_regressor import TransformerRegressor
        d_model = model_params.get("d_model", 128)
        nhead = model_params.get("nhead", 8)
        num_encoder_layers = model_params.get("num_encoder_layers", 4)
        dropout = model_params.get("dropout", 0.1)
        model = TransformerRegressor(input_dim=input_dim, output_dim=output_dim, d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, dropout=dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load weights
    best_model_path = model_dir / "best_model.pt"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        raise FileNotFoundError(f"Model file not found: {best_model_path}")

    return model.to(device)


def predict_torch(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Get predictions from a PyTorch model."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for features, _ in loader:
            features = features.to(device)
            outputs = model(features)
            predictions.append(outputs.cpu().numpy())
    return np.concatenate(predictions, axis=0)


def predict_xgboost(model_dir: Path, test_features: np.ndarray, output_columns: List[str]) -> np.ndarray:
    """Get predictions from XGBoost models."""
    predictions = []
    for target in output_columns:
        model_path = model_dir / f"best_model_{target}.xgb"
        if not model_path.exists():
            raise FileNotFoundError(f"XGBoost model not found: {model_path}")
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        dtest = xgb.DMatrix(test_features)
        pred = booster.predict(dtest)
        predictions.append(pred)
    return np.column_stack(predictions)


def predict_lightgbm(model_dir: Path, test_features: np.ndarray, output_columns: List[str]) -> np.ndarray:
    """Get predictions from LightGBM models."""
    predictions = []
    for target in output_columns:
        model_path = model_dir / f"best_model_{target}.lgb"
        if not model_path.exists():
            raise FileNotFoundError(f"LightGBM model not found: {model_path}")
        booster = lgb.Booster(model_file=str(model_path))
        pred = booster.predict(test_features)
        predictions.append(pred)
    return np.column_stack(predictions)


def predict_catboost(model_dir: Path, test_features: np.ndarray, output_columns: List[str]) -> np.ndarray:
    """Get predictions from CatBoost models."""
    predictions = []
    for target in output_columns:
        model_path = model_dir / f"best_model_{target}.cbm"
        if not model_path.exists():
            raise FileNotFoundError(f"CatBoost model not found: {model_path}")
        model = CatBoostRegressor()
        model.load_model(str(model_path))
        pred = model.predict(test_features)
        predictions.append(pred)
    return np.column_stack(predictions)


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)

    # Load result config
    result = load_result_toml(model_dir)
    model_type = result.get("model_type", "MLP")
    model_format = result.get("model_format", "torch")
    output_columns = result.get("output_columns", [])

    # Load config for data paths
    config = load_config(model_dir)
    data_params = config.get("data", {})
    input_columns = data_params.get("input_columns", [])

    # Determine test CSV
    test_csv = args.test_csv
    if not test_csv:
        test_csv = str(REPO_ROOT / data_params.get("test_csv", "data/test.csv"))

    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else model_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load scalers
    scalers_mean, scalers_std = load_scalers(model_dir)

    # Load test data
    df_test = pl.read_csv(test_csv, encoding="utf-8-sig")
    columns = df_test.columns

    # Check for timestamp column
    ts_col = "Date, Time" if "Date, Time" in columns else None

    df_test = load_time_series(Path(test_csv), ts_col)
    columns = df_test.columns

    feature_indices, target_indices = get_feature_and_target_indices(columns, input_columns, output_columns)

    values_test = df_test.to_numpy().astype(np.float32)
    scaled_test = scale_values(values_test, scalers_mean, scalers_std)

    history_length = config.get("model", {}).get("history_length", 1)

    test_bundle = build_dataset_bundle(model_type, scaled_test, feature_indices, target_indices, history_length)
    test_loader = DataLoader(test_bundle.dataset, batch_size=256, shuffle=False)

    # Get predictions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_format == "torch":
        model = load_torch_model(model_dir, model_type, test_bundle.input_dim, len(output_columns), device)
        y_pred = predict_torch(model, test_loader, device)
    elif model_format == "xgboost":
        y_pred = predict_xgboost(model_dir, test_bundle.features, output_columns)
    elif model_format == "lightgbm":
        y_pred = predict_lightgbm(model_dir, test_bundle.features, output_columns)
    elif model_format == "catboost":
        y_pred = predict_catboost(model_dir, test_bundle.features, output_columns)
    else:
        raise ValueError(f"Unknown model format: {model_format}")

    y_true = test_bundle.targets

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)

    print(f"Test Results for {model_type}:")
    print(f"  MSE:  {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  R2:   {metrics['r2']:.6f}")

    # Save metrics
    metrics_path = output_dir / "test_metrics.csv"
    with metrics_path.open("w") as f:
        f.write("metric,value\n")
        for k, v in metrics.items():
            f.write(f"{k},{v}\n")

    # Plot predictions
    plot_predictions(output_columns, y_true, y_pred, output_dir)

    # Save predictions
    pred_df = pl.DataFrame(y_pred, schema=output_columns)
    pred_df.write_csv(output_dir / "test_predictions.csv")

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()