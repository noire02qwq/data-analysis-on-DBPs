"""
Shared helpers for regression experiments: data loading, scaling, and dataset builders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TARGET_COLUMNS: List[str] = [
    "TRC-PPL1",
    "TRC-PPL2",
    "TOC-PPL1",
    "TOC-PPL2",
    "DOC-PPL1",
    "DOC-PPL2",
    "pH-PPL1",
    "pH-PPL2",
]

PPL_SUFFIXES = ("-PPL1", "-PPL2")


@dataclass(frozen=True)
class SplitBoundaries:
    train_end: int
    val_end: int
    test_end: int


class CurrentStepDataset(Dataset):
    """Per-step dataset used by MLP and XGBoost."""

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.valid_indices = np.arange(len(self.features), dtype=int)
        self.input_dim = self.features.shape[1]
        self.sequence_length: int | None = None

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.features[idx]), torch.from_numpy(self.targets[idx])


class HistoryFlattenDataset(Dataset):
    """Windowed dataset that flattens history_length steps of non-PPL features."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, history_length: int) -> None:
        if history_length < 1:
            raise ValueError("history_length must be >= 1 for MLP_WITH_HISTORY.")
        total_rows = features.shape[0]
        if total_rows < history_length:
            raise ValueError("Not enough samples to build the requested history window.")

        valid_indices = np.arange(history_length - 1, total_rows, dtype=int)
        stacked = [
            features[idx - history_length + 1 : idx + 1].reshape(-1) for idx in valid_indices
        ]
        self.features = np.stack(stacked).astype(np.float32)
        self.targets = targets[valid_indices].astype(np.float32)
        self.valid_indices = valid_indices
        self.input_dim = self.features.shape[1]
        self.sequence_length: int | None = None

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.features[idx]), torch.from_numpy(self.targets[idx])


class SequenceDataset(Dataset):
    """Sequence dataset feeding history_length steps into LSTM/RNN models."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, history_length: int) -> None:
        if history_length < 1:
            raise ValueError("history_length must be >= 1 for sequential models.")
        total_rows = features.shape[0]
        if total_rows < history_length:
            raise ValueError("Not enough samples to build the requested history window.")

        valid_indices = np.arange(history_length - 1, total_rows, dtype=int)
        windows = [features[idx - history_length + 1 : idx + 1] for idx in valid_indices]
        self.features = np.stack(windows).astype(np.float32)
        self.targets = targets[valid_indices].astype(np.float32)
        self.valid_indices = valid_indices
        self.input_dim = self.features.shape[2]
        self.sequence_length = history_length

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


def load_time_series(csv_path: Path, timestamp_column: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if timestamp_column not in df.columns:
        raise KeyError(f"Timestamp column '{timestamp_column}' not found in {csv_path}.")

    df = df.copy()
    df["_timestamp"] = pd.to_datetime(df[timestamp_column])
    df = df.sort_values("_timestamp").reset_index(drop=True)
    df["minutes_since_start"] = (df["_timestamp"] - df["_timestamp"].min()).dt.total_seconds() / 60.0
    df = df.drop(columns=[timestamp_column, "_timestamp"])

    numeric_df = df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    return numeric_df


def is_ppl_column(name: str) -> bool:
    return name.endswith(PPL_SUFFIXES)


def get_feature_and_target_indices(columns: Sequence[str]) -> Tuple[List[int], List[int]]:
    column_to_idx = {col: idx for idx, col in enumerate(columns)}
    target_indices: List[int] = []
    for col in TARGET_COLUMNS:
        if col not in column_to_idx:
            missing = set(TARGET_COLUMNS) - set(column_to_idx)
            raise ValueError(f"Missing target columns: {missing}")
        target_indices.append(column_to_idx[col])

    feature_indices = [idx for idx, name in enumerate(columns) if not is_ppl_column(name)]
    if not feature_indices:
        raise ValueError("No non-PPL columns remain for features.")
    return feature_indices, target_indices


def compute_split_boundaries(n_rows: int) -> SplitBoundaries:
    train_end = int(n_rows * 0.7)
    val_end = train_end + int(n_rows * 0.15)
    return SplitBoundaries(train_end=train_end, val_end=val_end, test_end=n_rows)


def compute_scalers(values: np.ndarray, train_end: int) -> Tuple[np.ndarray, np.ndarray]:
    train_slice = values[:train_end]
    mean = train_slice.mean(axis=0)
    std = train_slice.std(axis=0)
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
) -> DatasetBundle:
    non_ppl = scaled_values[:, feature_indices]
    target_data = scaled_values[:, target_indices]
    model = model_type.upper()

    if model in {"MLP", "XGBOOST"}:
        dataset: Dataset = CurrentStepDataset(non_ppl, target_data)
    elif model == "MLP_WITH_HISTORY":
        dataset = HistoryFlattenDataset(non_ppl, target_data, history_length)
    elif model in {"LSTM", "RNN"}:
        dataset = SequenceDataset(non_ppl, target_data, history_length)
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'.")

    return DatasetBundle(
        dataset=dataset,
        features=dataset.features,
        targets=dataset.targets,
        valid_indices=dataset.valid_indices,
        input_dim=dataset.input_dim,
        sequence_length=getattr(dataset, "sequence_length", None),
    )


def subset_indices(valid_indices: np.ndarray, boundaries: SplitBoundaries) -> Dict[str, np.ndarray]:
    train_mask = valid_indices < boundaries.train_end
    val_mask = (valid_indices >= boundaries.train_end) & (valid_indices < boundaries.val_end)
    test_mask = valid_indices >= boundaries.val_end

    splits = {
        "train": np.nonzero(train_mask)[0],
        "val": np.nonzero(val_mask)[0],
        "test": np.nonzero(test_mask)[0],
    }
    for name, idx in splits.items():
        if idx.size == 0:
            raise ValueError(f"{name} split is empty. Adjust history length or split ratios.")
    return splits
