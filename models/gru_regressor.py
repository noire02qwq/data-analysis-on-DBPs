"""
GRU-based regressor mirroring the LSTM architecture.
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn


class GRURegressor(nn.Module):
    """Stacked GRU regressor with an optional fully connected head."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 192,
        num_layers: int = 2,
        dropout: float = 0.0,
        fc_dim: int | None = None,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        head_layers: List[nn.Module] = []
        if fc_dim is not None:
            head_layers.append(nn.Linear(hidden_size, fc_dim))
            head_layers.append(nn.ReLU())
            head_layers.append(nn.Dropout(dropout) if dropout > 0 else nn.Identity())
            head_layers.append(nn.Linear(fc_dim, output_dim))
        else:
            head_layers.append(nn.Linear(hidden_size, output_dim))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.gru(x)
        last_hidden = outputs[:, -1, :]
        return self.head(last_hidden)