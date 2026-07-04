"""Reverse-time SDE sampler for VP score models."""

from __future__ import annotations

import numpy as np
import torch

from src.integrators import euler_maruyama_step
from src.sampling.utils import (
    default_device,
    reverse_sde_forward_drift,
    reverse_time_to_forward_time,
)


def sample_reverse_sde(
    wrapper,
    process,
    n_samples: int = 2000,
    n_steps: int = 500,
    seed: int = 0,
    device: torch.device | None = None,
    t_min: float = 1e-3,
    return_trajectory: bool = False,
    record_every: int = 1,
) -> torch.Tensor | tuple[torch.Tensor, np.ndarray]:
    """
    Generate samples by integrating Song et al.'s reverse-time SDE.

    Starts from x(t=1) ~ N(0,I) and advances positive reverse time s from
    0 to 1, where forward diffusion time is t=1-s. Euler-Maruyama is called
    with positive dt, using reverse-time drift:
        d x / d s = -[f(x,t) - g(t)^2 score_t(x)],  t = 1 - s.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    device = default_device() if device is None else device
    torch.manual_seed(seed)

    x = torch.randn(n_samples, 2, device=device)
    dt = 1.0 / n_steps
    trajectory = []
    if return_trajectory:
        trajectory.append(x.detach().cpu().numpy())

    def drift_s(x_batch: torch.Tensor, s_batch: torch.Tensor) -> torch.Tensor:
        t_batch = reverse_time_to_forward_time(s_batch, t_min)
        return -reverse_sde_forward_drift(wrapper, process, x_batch, t_batch)

    def diffusion_s(s_batch: torch.Tensor) -> torch.Tensor:
        t_batch = reverse_time_to_forward_time(s_batch, t_min)
        return process.diffusion(t_batch)

    with torch.no_grad():
        for step in range(n_steps):
            s = torch.full((n_samples,), step * dt, device=device)
            x = euler_maruyama_step(x, s, dt, drift_s, diffusion_s)
            if return_trajectory and ((step + 1) % record_every == 0 or step == n_steps - 1):
                trajectory.append(x.detach().cpu().numpy())

    if return_trajectory:
        return x, np.stack(trajectory, axis=0)
    return x
