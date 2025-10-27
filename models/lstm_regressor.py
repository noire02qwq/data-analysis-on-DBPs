"""
LSTM-based regression network used in the DBPS experiments.
"""

from __future__ import annotations

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    """LSTM that consumes per-step inputs and predicts the final step outputs."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        fc_dim: int | None = None,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        head_dim = fc_dim or hidden_size
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(x)
        last_hidden = outputs[:, -1, :]
        return self.fc(last_hidden)
