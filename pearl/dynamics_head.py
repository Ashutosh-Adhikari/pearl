# Copyright 2026 Ashutosh Adhikari and Mirella Lapata.
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn


class DynamicsHead(nn.Module):
    """Next-latent predictor used by the NextLat loss.

    Takes ``[h_t ; e_{t+1}]`` (hidden state at step t and the embedding of the
    next input token) and predicts ``h_{t+1}`` through a residual MLP.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ln = nn.LayerNorm(2 * hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., : -self.hidden_size] + self.mlp(self.ln(x))
