"""
Shared helpers for regression experiments: data loading, scaling, and dataset builders.
Uses Polars instead of Pandas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


class CurrentStepDataset(Dataset):
    """Per-step dataset used by MLP and XGBoost."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        base_targets: np.ndarray | None = None,
    ) -> None:
        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.valid_indices = np.arange(len(self.features), dtype=int)
        self.input_dim = self.features.shape[1]
        self.sequence_length: int | None = None
        if base_targets is not None:
            if base_targets.shape != targets.shape:
                raise ValueError("base_targets must match targets shape.")
            self.base_targets = base_targets.astype(np.float32)
        else:
            self.base_targets = None

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.features[idx]), torch.from_numpy(self.targets[idx])


class SequenceDataset(Dataset):
    """Sequence dataset feeding history_length steps into LSTM/RNN models."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        history_length: int,
        base_targets: np.ndarray | None = None,
    ) -> None:
        if history_length < 1:
            raise ValueError("history_length must be >= 1 for sequential models.")
        total_rows = features.shape[0]
        if total_rows < history_length:
            raise ValueError("Not enough samples to build the requested history window.")

        history_length = int(history_length)
        valid_indices = np.arange(history_length - 1, total_rows, dtype=int)
        windows = [features[idx - history_length + 1 : idx + 1] for idx in valid_indices]
        self.features = np.stack(windows).astype(np.float32)
        self.targets = targets[valid_indices].astype(np.float32)
        self.valid_indices = valid_indices
        self.input_dim = self.features.shape[2]
        self.sequence_length = history_length
        if base_targets is not None:
            if base_targets.shape != targets.shape:
                raise ValueError("base_targets must match targets shape.")
            self.base_targets = base_targets[valid_indices].astype(np.float32)
        else:
            self.base_targets = None

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.features[idx]), torch.from_numpy(self.targets[idx])


@dataclass
class DatasetBundle:
    dataset: Dataset
    features: np.ndarray
    targets: np.ndarray
    valid_indices: np.ndarray
    input_dim: int
    sequence_length: int | None
    base_targets: np.ndarray | None


def load_time_series(csv_path: Path, timestamp_column: str) -> pl.DataFrame:
    df = pl.read_csv(csv_path, encoding="utf-8-sig")
    if timestamp_column not in df.columns:
        raise KeyError(f"Timestamp column '{timestamp_column}' not found in {csv_path}.")

    df = df.with_columns([
        pl.col(timestamp_column).str.to_datetime().alias("_timestamp")
    ])

    df = df.sort("_timestamp")
    min_ts = df["_timestamp"][0]
    df = df.with_columns([
        ((pl.col("_timestamp") - min_ts).dt.total_seconds() / 60.0).alias("minutes_since_start")
    ])

    # Drop timestamp columns and keep numeric
    columns_to_keep = [c for c in df.columns if c not in [timestamp_column, "_timestamp"]]
    df = df.select(columns_to_keep)

    # Convert to numeric, coercing errors to null, then drop nulls
    numeric_cols = []
    for col in df.columns:
        try:
            numeric_cols.append(pl.col(col).cast(pl.Float64))
        except:
            numeric_cols.append(pl.col(col))
    df = df.with_columns(numeric_cols)
    df = df.drop_nulls()

    return df


def get_feature_and_target_indices(
    columns: List[str],
    input_columns: List[str],
    output_columns: List[str]
) -> Tuple[List[int], List[int]]:

    column_to_idx = {col: idx for idx, col in enumerate(columns)}

    target_indices: List[int] = []
    for col in output_columns:
        if col not in column_to_idx:
            raise ValueError(f"Missing output column: {col}")
        target_indices.append(column_to_idx[col])

    feature_indices: List[int] = []
    for col in input_columns:
        if col not in column_to_idx:
            raise ValueError(f"Missing input column: {col}")
        feature_indices.append(column_to_idx[col])

    if not feature_indices:
        raise ValueError("No input columns provided.")
    if not target_indices:
        raise ValueError("No output columns provided.")

    return feature_indices, target_indices


def compute_scalers(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and std on the given values (usually just the training set)."""
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def scale_values(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (values - mean) / std


def build_dataset_bundle(
    model_type: str,
    scaled_values: np.ndarray,
    feature_indices: List[int],
    target_indices: List[int],
    history_length: int,
    base_targets: np.ndarray | None = None,
) -> DatasetBundle:
    non_ppl = scaled_values[:, feature_indices]
    target_data = scaled_values[:, target_indices]
    model = model_type.upper()

    if model in {"MLP", "XGBOOST", "LIGHTGBM", "CATBOOST"}:
        dataset: Dataset = CurrentStepDataset(non_ppl, target_data, base_targets=base_targets)
    elif model in {"LSTM", "RNN", "GRU", "TRANSFORMER", "MAMBA"}:
        dataset = SequenceDataset(non_ppl, target_data, history_length, base_targets=base_targets)
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'.")

    return DatasetBundle(
        dataset=dataset,
        features=dataset.features,
        targets=dataset.targets,
        valid_indices=dataset.valid_indices,
        input_dim=dataset.input_dim,
        sequence_length=getattr(dataset, "sequence_length", None),
        base_targets=getattr(dataset, "base_targets", None),
    )