#!/usr/bin/env python3
"""
Enhanced test script with comprehensive output:
- True vs Predicted plots for each output variable
- y=x scatter plots with R² for each output variable
- Prediction comparison table
- Comprehensive test metrics
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
    parser = argparse.ArgumentParser(description="Enhanced test script.")
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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {"mse": float(mse), "rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


def compute_metrics_per_target(y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str]) -> List[Dict]:
    """Compute metrics for each target separately."""
    results = []
    for i, name in enumerate(target_names):
        true_i = y_true[:, i]
        pred_i = y_pred[:, i]

        mse = np.mean((true_i - pred_i) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(true_i - pred_i))

        ss_res = np.sum((true_i - pred_i) ** 2)
        ss_tot = np.sum((true_i - np.mean(true_i)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        results.append({
            "target": name,
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
        })
    return results


def plot_predictions_per_target(
    target_names: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path
) -> None:
    """Plot predictions vs true for each target."""
    n_targets = len(target_names)
    n_cols = min(3, n_targets)
    n_rows = (n_targets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_targets == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, name in enumerate(target_names):
        ax = axes[idx]
        ax.plot(y_true[:, idx], label='True', linewidth=1.5, alpha=0.8)
        ax.plot(y_pred[:, idx], label='Predicted', linewidth=1.5, alpha=0.8, linestyle='--')
        ax.set_title(f'{name}', fontsize=11)
        ax.set_xlabel('Sample')
        ax.set_ylabel('Value')
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)

    for idx in range(n_targets, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "all_pred_vs_true.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Also save individual plots
    for idx, name in enumerate(target_names):
        plt.figure(figsize=(10, 4))
        plt.plot(y_true[:, idx], label='True', linewidth=1.5, alpha=0.8)
        plt.plot(y_pred[:, idx], label='Predicted', linewidth=1.5, alpha=0.8, linestyle='--')
        plt.title(f'{name} - True vs Predicted', fontsize=12)
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        safe_name = name.replace("/", "_").replace(" ", "_")
        plt.savefig(output_dir / f"{safe_name}_pred_vs_true.png", dpi=150)
        plt.close()


def plot_yx_scatter_per_target(
    target_names: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path
) -> None:
    """Plot y=x scatter with R² for each target."""
    n_targets = len(target_names)
    n_cols = min(3, n_targets)
    n_rows = (n_targets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_targets == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, name in enumerate(target_names):
        ax = axes[idx]
        true_i = y_true[:, idx]
        pred_i = y_pred[:, idx]

        # Calculate R²
        ss_res = np.sum((true_i - pred_i) ** 2)
        ss_tot = np.sum((true_i - np.mean(true_i)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        rmse = np.sqrt(np.mean((true_i - pred_i) ** 2))

        # Scatter
        ax.scatter(true_i, pred_i, c='blue', alpha=0.5, s=20, edgecolors='none')

        # y=x line
        min_val = min(true_i.min(), pred_i.min())
        max_val = max(true_i.max(), pred_i.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')

        ax.set_xlabel('True Values', fontsize=10)
        ax.set_ylabel('Predicted Values', fontsize=10)
        ax.set_title(f'{name}\nR² = {r2:.4f}, RMSE = {rmse:.4f}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_aspect('equal', adjustable='box')

    for idx in range(n_targets, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "all_yx_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Also save individual plots
    for idx, name in enumerate(target_names):
        true_i = y_true[:, idx]
        pred_i = y_pred[:, idx]

        ss_res = np.sum((true_i - pred_i) ** 2)
        ss_tot = np.sum((true_i - np.mean(true_i)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        rmse = np.sqrt(np.mean((true_i - pred_i) ** 2))

        plt.figure(figsize=(8, 7))
        plt.scatter(true_i, pred_i, c='blue', alpha=0.5, s=30, edgecolors='none')
        min_val = min(true_i.min(), pred_i.min())
        max_val = max(true_i.max(), pred_i.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
        plt.xlabel('True Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.title(f'{name}\nR² = {r2:.4f}, RMSE = {rmse:.4f}', fontsize=13)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()

        safe_name = name.replace("/", "_").replace(" ", "_")
        plt.savefig(output_dir / f"{safe_name}_yx_scatter.png", dpi=150)
        plt.close()


def load_torch_model(model_dir: Path, model_type: str, input_dim: int, output_dim: int, device: torch.device) -> nn.Module:
    """Load a PyTorch model from checkpoint."""
    config_path = model_dir / "config.toml"
    model_params = {}
    if config_path.exists():
        import tomli
        with config_path.open("rb") as f:
            config = tomli.load(f)
        model_params = config.get("model", {})

    if model_type == "MLP":
        hidden_layers = model_params.get("hidden_layers", [512, 256, 128])
        dropout = model_params.get("dropout", 0.0)
        model = MLPRegressor(input_dim=input_dim, output_dim=output_dim, hidden_layers=hidden_layers, dropout=dropout)
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
    elif model_type == "MAMBA":
        from models.mamba_regressor import MambaRegressor
        d_model = model_params.get("d_model", 128)
        n_layers = model_params.get("n_layers", 4)
        dropout = model_params.get("dropout", 0.1)
        model = MambaRegressor(input_dim=input_dim, output_dim=output_dim, d_model=d_model, n_layers=n_layers, dropout=dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    best_model_path = model_dir / "best_model.pt"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        raise FileNotFoundError(f"Model file not found: {best_model_path}")

    return model.to(device)


def predict_torch(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.no_grad():
        for features, _ in loader:
            features = features.to(device)
            outputs = model(features)
            predictions.append(outputs.cpu().numpy())
    return np.concatenate(predictions, axis=0)


def predict_xgboost(model_dir: Path, test_features: np.ndarray, output_columns: List[str]) -> np.ndarray:
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

    result = load_result_toml(model_dir)
    model_type = result.get("model_type", "MLP")
    model_format = result.get("model_format", "torch")
    output_columns = result.get("output_columns", [])

    config = load_config(model_dir)
    data_params = config.get("data", {})
    input_columns = data_params.get("input_columns", [])

    test_csv = args.test_csv
    if not test_csv:
        test_csv = str(REPO_ROOT / data_params.get("test_csv", "data/test.csv"))

    output_dir = Path(args.output_dir) if args.output_dir else model_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scalers_mean, scalers_std = load_scalers(model_dir)

    df_test = pl.read_csv(test_csv, encoding="utf-8-sig")
    columns = df_test.columns

    if "Date, Time" in columns:
        df_test = load_time_series(Path(test_csv), "Date, Time")
        columns = df_test.columns
    else:
        if "minutes_since_start" not in columns:
            df_test = df_test.with_columns([
                (pl.arange(0, df_test.height) * 5).alias("minutes_since_start")
            ])
        columns = df_test.columns

    model_columns = input_columns + output_columns

    missing_cols = [c for c in model_columns if c not in columns]
    if missing_cols:
        raise ValueError(f"Missing columns in test data: {missing_cols}. Available: {columns}")

    df_test = df_test.select(model_columns)
    columns = df_test.columns

    feature_indices, target_indices = get_feature_and_target_indices(columns, input_columns, output_columns)

    values_test = df_test.to_numpy().astype(np.float32)

    if scalers_mean.shape[0] != values_test.shape[1]:
        train_csv = REPO_ROOT / result.get("train_csv", config.get("data", {}).get("train_csv", "data/train.csv"))
        if not train_csv.exists():
            train_csv = REPO_ROOT / "data/train.csv"

        df_train = pl.read_csv(train_csv, encoding="utf-8-sig")
        full_columns = df_train.columns

        if "Date, Time" in full_columns:
            full_columns = [c for c in full_columns if c != "Date, Time"]

        indices = []
        for col in model_columns:
            if col in full_columns:
                indices.append(full_columns.index(col))
            else:
                raise ValueError(f"Model column '{col}' not found in training data columns")

        scalers_mean = scalers_mean[indices]
        scalers_std = scalers_std[indices]

    scaled_test = scale_values(values_test, scalers_mean, scalers_std)

    history_length = config.get("model", {}).get("history_length", 1)

    test_bundle = build_dataset_bundle(model_type, scaled_test, feature_indices, target_indices, history_length)
    test_loader = DataLoader(test_bundle.dataset, batch_size=256, shuffle=False)

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

    # Compute overall metrics
    overall_metrics = compute_metrics(y_true, y_pred)
    print(f"\nOverall Test Results for {model_type}:")
    print(f"  MSE:  {overall_metrics['mse']:.6f}")
    print(f"  RMSE: {overall_metrics['rmse']:.6f}")
    print(f"  MAE:  {overall_metrics['mae']:.6f}")
    print(f"  R2:   {overall_metrics['r2']:.6f}")

    # Compute per-target metrics
    per_target_metrics = compute_metrics_per_target(y_true, y_pred, output_columns)
    print(f"\nPer-Target Metrics:")
    for m in per_target_metrics:
        print(f"  {m['target']}: MSE={m['mse']:.6f}, RMSE={m['rmse']:.6f}, MAE={m['mae']:.6f}, R2={m['r2']:.6f}")

    # Save overall metrics
    metrics_path = output_dir / "test_metrics.csv"
    with metrics_path.open("w") as f:
        f.write("metric,value\n")
        for k, v in overall_metrics.items():
            f.write(f"{k},{v}\n")

    # Save per-target metrics
    per_target_path = output_dir / "test_metrics_per_target.csv"
    with per_target_path.open("w") as f:
        f.write("target,mse,rmse,mae,r2\n")
        for m in per_target_metrics:
            f.write(f"{m['target']},{m['mse']},{m['rmse']},{m['mae']},{m['r2']}\n")

    # Generate plots
    print("\nGenerating plots...")
    plot_predictions_per_target(output_columns, y_true, y_pred, output_dir)
    plot_yx_scatter_per_target(output_columns, y_true, y_pred, output_dir)

    # Save predictions comparison table
    n_samples = min(100, len(y_true))
    comp_df = pl.DataFrame({
        "sample_idx": list(range(n_samples)),
    })
    for i, col in enumerate(output_columns):
        comp_df = comp_df.with_columns([
            pl.lit(y_true[:n_samples, i]).alias(f"{col}_true"),
            pl.lit(y_pred[:n_samples, i]).alias(f"{col}_pred"),
        ])
    comp_df.write_csv(output_dir / "test_comparison.csv")

    print(f"\nResults saved to {output_dir}")
    print(f"  - test_metrics.csv (overall)")
    print(f"  - test_metrics_per_target.csv")
    print(f"  - test_comparison.csv")
    print(f"  - all_pred_vs_true.png")
    print(f"  - all_yx_scatter.png")
    for col in output_columns:
        safe_name = col.replace("/", "_").replace(" ", "_")
        print(f"  - {safe_name}_pred_vs_true.png")
        print(f"  - {safe_name}_yx_scatter.png")


if __name__ == "__main__":
    main()