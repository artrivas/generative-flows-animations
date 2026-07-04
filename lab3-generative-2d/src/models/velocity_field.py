"""MLP velocity field for Conditional Flow Matching on 2D toy data."""

import torch
import torch.nn as nn

from src.models.denoiser import sinusoidal_time_embedding


class VelocityField(nn.Module):
    """
    MLP: input = concat(x, time_embedding(t)) -> velocity in R^2.

    The Flow Matching convention used in this repo is NOTES.md section 8:
        t=0: base noise, t=1: data,
        x_t = (1-t) x_base + t x_data,
        target velocity = x_data - x_base.
    """

    def __init__(
        self,
        data_dim: int = 2,
        time_emb_dim: int = 32,
        hidden_dim: int = 128,
        n_layers: int = 4,
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        layers = []
        in_dim = data_dim + time_emb_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
            in_dim = hidden_dim
        layers += [nn.Linear(in_dim, data_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(t, self.time_emb_dim)
        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h)
