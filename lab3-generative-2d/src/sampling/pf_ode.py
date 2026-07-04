"""Probability Flow ODE sampler for VP score models."""

from __future__ import annotations

import numpy as np
import torch

from src.integrators import euler_step, heun_step
from src.sampling.utils import (
    default_device,
    probability_flow_drift,
    reverse_time_to_forward_time,
)


def sample_pf_ode(
    wrapper,
    process,
    n_samples: int = 2000,
    n_steps: int = 500,
    seed: int = 0,
    method: str = "heun",
    device: torch.device | None = None,
    t_min: float = 1e-3,
    return_trajectory: bool = False,
    record_every: int = 1,
) -> torch.Tensor | tuple[torch.Tensor, np.ndarray]:
    """
    Generate samples by integrating the Probability Flow ODE backward.

    NOTES.md section 7 / Song et al.:
        dx = [f(x,t) - 1/2 g(t)^2 score_t(x)] dt

    Sampling uses reverse time s in [0,1], so the positive-dt drift passed
    to Euler/Heun is the negative of the forward-time PF-ODE drift.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if method not in {"euler", "heun"}:
        raise ValueError(f"method must be 'euler' or 'heun', got {method!r}")
    device = default_device() if device is None else device
    torch.manual_seed(seed)

    x = torch.randn(n_samples, 2, device=device)
    dt = 1.0 / n_steps
    step_fn = heun_step if method == "heun" else euler_step
    trajectory = []
    if return_trajectory:
        trajectory.append(x.detach().cpu().numpy())

    def drift_s(x_batch: torch.Tensor, s_batch: torch.Tensor) -> torch.Tensor:
        t_batch = reverse_time_to_forward_time(s_batch, t_min)
        return -probability_flow_drift(wrapper, process, x_batch, t_batch)

    with torch.no_grad():
        for step in range(n_steps):
            s = torch.full((n_samples,), step * dt, device=device)
            x = step_fn(x, s, dt, drift_s)
            if return_trajectory and ((step + 1) % record_every == 0 or step == n_steps - 1):
                trajectory.append(x.detach().cpu().numpy())

    if return_trajectory:
        return x, np.stack(trajectory, axis=0)
    return x
