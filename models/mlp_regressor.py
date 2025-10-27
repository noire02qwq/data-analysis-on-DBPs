"""
Feed-forward regression network used in the DBPS experiments.
"""

from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


class MLPRegressor(nn.Module):
    """Simple feed-forward network for flattened time-window inputs."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: Iterable[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
