"""
Shared helpers for the regression experiments (data prep, datasets, constants).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TARGET_COLUMNS: List[str] = [
    "DO-PPL1",
    "DO-PPL2",
    "TOC-PPL1",
    "TOC-PPL2",
    "DOC-PPL1",
    "DOC-PPL2",
]


@dataclass(frozen=True)
class SplitBoundaries:
    train_end: int
    val_end: int
    test_end: int


def load_time_series(csv_path: Path, timestamp_column: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if timestamp_column not in df.columns:
        raise KeyError(f"Timestamp column '{timestamp_column}' not found in {csv_path}.")

    df = df.copy()
    df["_timestamp"] = pd.to_datetime(df[timestamp_column])
    df = df.sort_values("_timestamp").reset_index(drop=True)
    df["minutes_since_start"] = (df["_timestamp"] - df["_timestamp"].min()).dt.total_seconds() / 60.0
    df = df.drop(columns=[timestamp_column, "_timestamp"])

    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.dropna().reset_index(drop=True)
    return numeric_df


def get_column_indices(columns: Sequence[str]) -> Tuple[List[int], List[int]]:
    target_indices = [i for i, c in enumerate(columns) if c in TARGET_COLUMNS]
    if len(target_indices) != len(TARGET_COLUMNS):
        missing = set(TARGET_COLUMNS) - {columns[i] for i in target_indices}
        raise ValueError(f"Missing target columns: {missing}")
    non_target_indices = [i for i, _ in enumerate(columns) if i not in target_indices]
    return non_target_indices, target_indices


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


class MLPPastSequenceDataset(Dataset):
    """Dataset for the MLP experiment that flattens the look-back window."""

    def __init__(
        self,
        all_data: np.ndarray,
        non_target_data: np.ndarray,
        target_data: np.ndarray,
        history_length: int,
    ) -> None:
        if history_length < 1:
            raise ValueError("history_length must be >= 1 for the MLP dataset.")
        self.all_data = all_data
        self.non_target_data = non_target_data
        self.target_data = target_data
        self.history_length = history_length
        self.valid_indices = np.arange(history_length, len(all_data), dtype=int)
        self.input_dim = history_length * all_data.shape[1] + non_target_data.shape[1]

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        target_idx = self.valid_indices[idx]
        start_idx = target_idx - self.history_length
        past_window = self.all_data[start_idx:target_idx].reshape(-1)
        current_non_target = self.non_target_data[target_idx]
        features = np.concatenate([past_window, current_non_target]).astype(np.float32)
        label = self.target_data[target_idx].astype(np.float32)
        return torch.from_numpy(features), torch.from_numpy(label)


class LSTMSequenceDataset(Dataset):
    """Dataset feeding sequences to the LSTM architecture."""

    def __init__(
        self,
        non_target_data: np.ndarray,
        target_data: np.ndarray,
        history_length: int,
    ) -> None:
        if history_length < 1:
            raise ValueError("history_length must be >= 1 for the LSTM dataset.")
        self.non_target_data = non_target_data
        self.target_data = target_data
        self.history_length = history_length
        self.valid_indices = np.arange(history_length - 1, len(non_target_data), dtype=int)
        self.input_dim = non_target_data.shape[1] + target_data.shape[1]

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        target_idx = self.valid_indices[idx]
        start_idx = target_idx - self.history_length + 1
        if start_idx < 0:
            raise IndexError("Not enough history to build the requested sample.")
        seq_non_targets = self.non_target_data[start_idx : target_idx + 1]
        seq_prev_outputs = []
        for step in range(start_idx, target_idx + 1):
            prev_idx = step - 1
            if prev_idx < 0:
                prev_target = np.zeros(self.target_data.shape[1], dtype=np.float32)
            else:
                prev_target = self.target_data[prev_idx]
            seq_prev_outputs.append(prev_target)
        seq_prev_outputs = np.stack(seq_prev_outputs, axis=0)
        sequence = np.concatenate([seq_non_targets, seq_prev_outputs], axis=1).astype(np.float32)
        label = self.target_data[target_idx].astype(np.float32)
        return torch.from_numpy(sequence), torch.from_numpy(label)


def subset_indices(valid_indices: np.ndarray, boundary: SplitBoundaries) -> Dict[str, np.ndarray]:
    train_mask = valid_indices < boundary.train_end
    val_mask = (valid_indices >= boundary.train_end) & (valid_indices < boundary.val_end)
    test_mask = valid_indices >= boundary.val_end

    splits = {
        "train": np.nonzero(train_mask)[0],
        "val": np.nonzero(val_mask)[0],
        "test": np.nonzero(test_mask)[0],
    }
    for name, idx in splits.items():
        if idx.size == 0:
            raise ValueError(f"{name} split is empty. Adjust history length or split ratios.")
    return splits
