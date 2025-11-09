"""
PyTorch MLP architectures for the DBPS regression experiments.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
from torch import nn


class MLPRegressor(nn.Module):
    """Feed-forward regressor that consumes flattened feature vectors."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: Sequence[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not hidden_layers:
            raise ValueError("hidden_layers must contain at least one entry.")

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for idx, hidden_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_layers: Iterable[int],
    dropout: float,
) -> MLPRegressor:
    return MLPRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=list(hidden_layers),
        dropout=dropout,
    )
