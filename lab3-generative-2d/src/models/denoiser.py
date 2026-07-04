"""
Simple MLP denoiser with sinusoidal time embedding, for 2D toy data.

DenoiserWrapper converts the raw network output (eps or v prediction) into
the score, using the EXACT formulas from references/NOTES.md §5 (eps) and
§6 (v):

    eps->score:  NOTES.md §5:  score = -eps_theta(x_t,t) / sqrt(1 - alpha_bar_t)
    v->score:    NOTES.md §6:  score = -(sqrt(alpha_bar_t) v_theta(x_t,t)
                                          + sqrt(1-alpha_bar_t) x_t) / (1-alpha_bar_t)

The VP forward process's marginal_prob(x0, t) returns exactly
(mean=alpha(t) x0, std=sqrt(1-alpha(t)^2)) -- i.e. alpha(t) plays the role
of sqrt(alpha_bar_t) and std(t) plays the role of sqrt(1-alpha_bar_t) in
the formulas above (see NOTES.md §4). eps/v-prediction is only meaningful
under this VP-style kernel -- use VPSDE as the forward_process passed to
DenoiserWrapper.
"""

import math

import torch
import torch.nn as nn


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Transformer-style sinusoidal embedding. t: (batch,) -> (batch, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class Denoiser(nn.Module):
    """
    MLP: input = concat(x, sinusoidal_time_embedding(t)) -> R^{data_dim}.

    The network is parametrization-agnostic; whether its output means
    "eps" or "v" is a property of how it was trained (see train_score.py)
    and is only interpreted downstream by DenoiserWrapper.
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


def epsilon_to_score(eps_pred: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    score = -eps_pred / sqrt(1 - alpha_bar_t).   NOTES.md §5.
    `std` = marginal_prob's sigma_t = sqrt(1 - alpha_bar_t) under VP.
    """
    return -eps_pred / std


def v_to_score(
    v_pred: torch.Tensor, x_t: torch.Tensor, alpha: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """
    score = -(sqrt(alpha_bar_t) v_pred + sqrt(1-alpha_bar_t) x_t) / (1-alpha_bar_t).
    NOTES.md §6. `alpha` = sqrt(alpha_bar_t) (VP's alpha(t)); `std` = sqrt(1-alpha_bar_t).
    """
    return -(alpha * v_pred + std * x_t) / (std ** 2)


class DenoiserWrapper:
    """
    Wraps a raw Denoiser network + a VP forward_process to produce the
    score, given the parametrization ("eps" or "v") the network was
    trained with.
    """

    def __init__(self, model: Denoiser, forward_process, param: str = "eps"):
        if param not in ("eps", "v"):
            raise ValueError(f"param must be 'eps' or 'v', got {param!r}")
        self.model = model
        self.forward_process = forward_process
        self.param = param

    def _alpha_and_std(self, x_ref: torch.Tensor, t: torch.Tensor):
        """
        alpha(t), std(t) via marginal_prob with x0=ones (so mean = alpha(t)
        * 1 = alpha(t) directly, avoiding division by a possibly-zero x0).
        """
        ones = torch.ones_like(x_ref)
        alpha, std = self.forward_process.marginal_prob(ones, t)
        return alpha, std

    def raw_output(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(x_t, t)

    def score(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raw = self.raw_output(x_t, t)
        alpha, std = self._alpha_and_std(x_t, t)
        if self.param == "eps":
            return epsilon_to_score(raw, std)
        return v_to_score(raw, x_t, alpha, std)
