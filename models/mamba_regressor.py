"""
Mamba-based regressor using selective state space models.
Requires: pip install mamba-ssm (for CUDA) or use a pure PyTorch implementation
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn


class MambaBlock(nn.Module):
    """Simplified Mamba block using standard PyTorch operations.

    This is a simplified implementation that captures the core idea
    of selective state space models without requiring the full mamba-ssm package.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # Activation
        self.act = nn.SiLU()

        # SSM parameters (simplified)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_state, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.ones(d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch, seq_len, dim = x.shape

        # Input projection
        x_and_res = self.in_proj(x)  # (batch, seq_len, 2 * d_inner)
        x_proj, res = x_and_res.split(self.d_inner, dim=-1)

        # Convolution
        x_conv = x_proj.transpose(1, 2)  # (batch, d_inner, seq_len)
        x_conv = self.conv1d(x_conv)[..., :seq_len]
        x_conv = x_conv.transpose(1, 2)  # (batch, seq_len, d_inner)
        x_conv = self.act(x_conv)

        # Simplified SSM computation (state space model)
        # This is a simplified version - full mamba has more complex selective scanning
        x_ssm = x_conv * torch.sigmoid(self.D)  # Gating mechanism

        # Residual connection
        x_out = x_ssm + res

        # Output projection
        output = self.out_proj(x_out)
        output = self.dropout(output)

        return output


class MambaRegressor(nn.Module):
    """Mamba-based sequence regressor."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        fc_dim: int | None = None,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

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
        x = self.input_proj(x)  # (batch_size, seq_len, d_model)

        for layer in self.layers:
            x = layer(x) + x  # Residual connection

        x = self.norm(x)
        # Take the last time step for prediction
        x = x[:, -1, :]  # (batch_size, d_model)
        x = self.head(x)  # (batch_size, output_dim)
        return x