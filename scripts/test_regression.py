#!/usr/bin/env python3
"""
Evaluate trained regression models (PyTorch or XGBoost) and generate prediction plots.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import LSTMRegressor, MLPRegressor, RNNRegressor, XGBoostRegressor  # noqa: E402
from scripts.regression_utils import (  # noqa: E402
    DatasetBundle,
    SplitBoundaries,
    build_dataset_bundle,
    load_time_series,
    scale_values,
    subset_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test trained regression models.")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing metadata.json and scalers.npz.",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Optional CSV override (defaults to metadata.data_csv).",
    )
    return parser.parse_args()


def load_metadata(model_dir: Path) -> Dict[str, object]:
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {model_dir}")
    with metadata_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_artifact(path_str: str, model_dir: Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        fallback = model_dir / Path(path_str).name
        if fallback.exists():
            return fallback
        raise FileNotFoundError(f"Artifact not found at {path} or {fallback}")
    return path


def plot_predictions(target_names: List[str], y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> None:
    for idx, name in enumerate(target_names):
        plt.figure(figsize=(8, 4))
        plt.plot(y_true[:, idx], label="True")
        plt.plot(y_pred[:, idx], label="Predicted")
        plt.title(f"{name} - True vs Predicted")
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        safe_name = name.replace("/", "_").replace(" ", "_")
        plt.savefig(output_dir / f"{safe_name}_pred_vs_true.png")
        plt.close()


def instantiate_torch_model(metadata: Dict[str, object], input_dim: int, output_dim: int) -> torch.nn.Module:
    model_type = metadata["model_type"]
    params = metadata["model_params"]
    sequence_length = metadata.get("sequence_length")
    if model_type == "MLP":
        return MLPRegressor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=params["hidden_layers"],
            dropout=float(params.get("dropout", 0.0)),
        )
    if model_type == "MLP_WITH_HISTORY":
        return MLPRegressor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=params["hidden_layers"],
            dropout=float(params.get("dropout", 0.0)),
        )
    if model_type == "LSTM":
        return LSTMRegressor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=int(params.get("units", 192)),
            num_layers=int(params.get("num_layers", 2)),
            dropout=float(params.get("dropout", 0.0)),
            fc_dim=params.get("fc_dim"),
        )
    if model_type == "RNN":
        return RNNRegressor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=int(params.get("units", 160)),
            num_layers=int(params.get("num_layers", 2)),
            dropout=float(params.get("dropout", 0.0)),
            fc_dim=params.get("fc_dim"),
        )
    raise ValueError(f"Unsupported torch model type '{model_type}'.")


def prepare_dataset(metadata: Dict[str, object], scaled_values: np.ndarray) -> Tuple[DatasetBundle, SplitBoundaries, List[int]]:
    columns: List[str] = metadata["columns"]  # type: ignore[assignment]
    feature_columns: List[str] = metadata["feature_columns"]  # type: ignore[assignment]
    target_columns: List[str] = metadata["target_columns"]  # type: ignore[assignment]
    column_to_idx = {col: idx for idx, col in enumerate(columns)}
    feature_indices = [column_to_idx[col] for col in feature_columns]
    target_indices = [column_to_idx[col] for col in target_columns]

    bundle = build_dataset_bundle(
        metadata["model_type"],
        scaled_values,
        feature_indices,
        target_indices,
        metadata["model_params"]["history_length"],
    )
    boundary_values = metadata["split_boundaries"]
    boundaries = SplitBoundaries(
        train_end=int(boundary_values["train_end"]),
        val_end=int(boundary_values["val_end"]),
        test_end=int(boundary_values["test_end"]),
    )
    return bundle, boundaries, target_indices


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    metadata = load_metadata(model_dir)

    data_path = Path(args.data) if args.data else Path(metadata["data_csv"])
    if not data_path.is_absolute():
        data_path = (REPO_ROOT / data_path).resolve()
    timestamp_column = metadata["timestamp_column"]

    df = load_time_series(data_path, timestamp_column)
    df = df[metadata["columns"]]

    scalers = np.load(model_dir / "scalers.npz")
    mean = scalers["mean"]
    std = scalers["std"]
    scaled_values = scale_values(df.to_numpy(dtype=np.float32), mean, std)

    dataset_bundle, boundaries, target_indices = prepare_dataset(metadata, scaled_values)
    splits = subset_indices(dataset_bundle.valid_indices, boundaries)
    test_indices = splits["test"]
    if test_indices.size == 0:
        raise RuntimeError("Test split is empty; cannot evaluate.")

    target_names: List[str] = metadata["target_columns"]  # type: ignore[assignment]

    if metadata["model_format"] == "xgboost":
        best_dir = resolve_artifact(metadata["model_files"]["best"], model_dir)
        xgb_model = XGBoostRegressor.load(best_dir, target_names)
        test_features = dataset_bundle.features[test_indices]
        y_pred_scaled = xgb_model.predict(test_features)
        y_true_scaled = dataset_bundle.targets[test_indices]
    else:
        dataset = dataset_bundle.dataset
        test_subset = Subset(dataset, test_indices)
        loader = DataLoader(test_subset, batch_size=256, shuffle=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = instantiate_torch_model(metadata, dataset_bundle.input_dim, len(target_names)).to(device)
        best_path = resolve_artifact(metadata["model_files"]["best"], model_dir)
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model_state"])
        model.eval()

        preds = []
        trues = []
        with torch.no_grad():
            for features, targets in loader:
                features = features.to(device)
                outputs = model(features)
                preds.append(outputs.cpu().numpy())
                trues.append(targets.numpy())
        y_pred_scaled = np.concatenate(preds, axis=0)
        y_true_scaled = np.concatenate(trues, axis=0)

    target_mean = mean[target_indices]
    target_std = std[target_indices]
    y_pred = y_pred_scaled * target_std + target_mean
    y_true = y_true_scaled * target_std + target_mean

    plot_predictions(target_names, y_true, y_pred, model_dir)

    result_path = model_dir / "test_predictions.csv"
    columns = []
    payload = []
    for idx, name in enumerate(target_names):
        columns.extend([f"{name}_true", f"{name}_pred"])
        payload.append(y_true[:, idx])
        payload.append(y_pred[:, idx])
    stacked = np.column_stack(payload)
    np.savetxt(result_path, stacked, delimiter=",", header=",".join(columns), comments="", fmt="%.6f")

    mse = np.mean((y_pred - y_true) ** 2, axis=0)
    for name, value in zip(target_names, mse):
        print(f"{name} MSE: {value:.6f}")
    print(f"Saved prediction plots and CSV to {model_dir}")


if __name__ == "__main__":
    main()
