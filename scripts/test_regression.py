#!/usr/bin/env python3
"""
Evaluate trained regression models and generate prediction plots.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import LSTMRegressor, MLPRegressor
from scripts.regression_utils import (
    LSTMSequenceDataset,
    MLPPastSequenceDataset,
    SplitBoundaries,
    load_time_series,
    scale_values,
    subset_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test trained regression models.")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing metadata.json, best_model.pt, and scalers.npz.",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Optional CSV file to evaluate on (defaults to the file recorded in metadata).",
    )
    return parser.parse_args()


def load_metadata(model_dir: Path) -> Dict[str, object]:
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {model_dir}")
    with metadata_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_dataset_from_metadata(
    df,
    metadata: Dict[str, object],
    scaled_values: np.ndarray,
) -> tuple[torch.utils.data.Dataset, SplitBoundaries, List[int], List[int]]:
    columns = metadata["columns"]
    df = df[columns]
    non_target_cols = metadata["non_target_columns"]
    target_cols = metadata["target_columns"]

    col_to_idx = {col: idx for idx, col in enumerate(columns)}
    non_target_indices = [col_to_idx[col] for col in non_target_cols]
    target_indices = [col_to_idx[col] for col in target_cols]

    model_type = metadata["model_type"]
    model_params = metadata["model_params"]
    history_length = int(model_params["history_length"])

    non_target_data = scaled_values[:, non_target_indices]
    target_data = scaled_values[:, target_indices]

    if model_type == "MLP":
        dataset = MLPPastSequenceDataset(
            scaled_values,
            non_target_data,
            target_data,
            history_length=history_length,
        )
    else:
        dataset = LSTMSequenceDataset(
            non_target_data,
            target_data,
            history_length=history_length,
        )

    boundary = SplitBoundaries(
        train_end=int(metadata["split_boundaries"]["train_end"]),
        val_end=int(metadata["split_boundaries"]["val_end"]),
        test_end=int(metadata["split_boundaries"]["test_end"]),
    )

    return dataset, boundary, non_target_indices, target_indices


def instantiate_model(metadata: Dict[str, object], input_dim: int, output_dim: int) -> torch.nn.Module:
    model_type = metadata["model_type"]
    params = metadata["model_params"]

    if model_type == "MLP":
        return MLPRegressor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=params["hidden_layers"],
            dropout=float(params.get("dropout", 0.0)),
        )
    return LSTMRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_size=int(params.get("hidden_size", 128)),
        num_layers=int(params.get("num_layers", 2)),
        dropout=float(params.get("dropout", 0.0)),
        fc_dim=params.get("fc_dim"),
    )


def plot_predictions(
    target_names: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
) -> None:
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


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    metadata = load_metadata(model_dir)

    data_path = Path(args.data) if args.data else Path(metadata["data_csv"])
    timestamp_column = metadata["timestamp_column"]

    df = load_time_series(data_path, timestamp_column)
    df = df[metadata["columns"]]

    scalers = np.load(model_dir / "scalers.npz")
    mean = scalers["mean"]
    std = scalers["std"]
    scaled_values = scale_values(df.to_numpy(dtype=np.float32), mean, std)

    dataset, boundary, non_target_indices, target_indices = build_dataset_from_metadata(
        df, metadata, scaled_values
    )

    splits = subset_indices(dataset.valid_indices, boundary)
    test_subset = Subset(dataset, splits["test"])
    if len(test_subset) == 0:
        raise RuntimeError("Test split is empty; cannot evaluate.")

    batch_size = 256
    loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = dataset.input_dim
    output_dim = len(target_indices)
    model = instantiate_model(metadata, input_dim, output_dim).to(device)

    checkpoint_path = model_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"best_model.pt not found in {model_dir}")
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    preds = []
    trues = []
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            outputs = model(features)
            preds.append(outputs.cpu().numpy())
            trues.append(targets.cpu().numpy())

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)

    target_mean = mean[target_indices]
    target_std = std[target_indices]
    y_pred_real = y_pred * target_std + target_mean
    y_true_real = y_true * target_std + target_mean

    plot_predictions(metadata["target_columns"], y_true_real, y_pred_real, model_dir)

    result_path = model_dir / "test_predictions.csv"
    columns = []
    data = []
    for idx, name in enumerate(metadata["target_columns"]):
        columns.extend([f"{name}_true", f"{name}_pred"])
        data.append(y_true_real[:, idx])
        data.append(y_pred_real[:, idx])
    stacked = np.column_stack(data)
    header = ",".join(columns)
    np.savetxt(result_path, stacked, delimiter=",", header=header, comments="", fmt="%.6f")

    mse = np.mean((y_pred_real - y_true_real) ** 2, axis=0)
    for name, value in zip(metadata["target_columns"], mse):
        print(f"{name} MSE: {value:.6f}")
    print(f"Saved prediction plots and CSV to {model_dir}")


if __name__ == "__main__":
    main()
