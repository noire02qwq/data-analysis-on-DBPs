"""
Transformer-based regressor using encoder-only architecture.
"""

from __future__ import annotations

import math
from typing import List

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Standard positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerRegressor(nn.Module):
    """Encoder-only Transformer regressor."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        fc_dim: int | None = None,
    ) -> None:
        super().__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # Pooling layer - take the last time step
        head_layers: List[nn.Module] = []
        if fc_dim is not None:
            head_layers.append(nn.Linear(d_model, fc_dim))
            head_layers.append(nn.ReLU())
            head_layers.append(nn.Dropout(dropout) if dropout > 0 else nn.Identity())
            head_layers.append(nn.Linear(fc_dim, output_dim))
        else:
            head_layers.append(nn.Linear(d_model, output_dim))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        x = self.pos_encoding(x)      # (batch_size, seq_len, d_model)
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        # Take the last time step for prediction
        x = x[:, -1, :]               # (batch_size, d_model)
        x = self.head(x)              # (batch_size, output_dim)
        return x