"""ODE sampler for trained Flow Matching velocity fields."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.integrators import euler_step, heun_step
from src.models.velocity_field import VelocityField
from src.sampling.utils import default_device


def load_velocity_model(checkpoint_path: str | Path, device: torch.device | None = None):
    """Load a step-9 Flow Matching checkpoint and return (model, config)."""
    device = default_device() if device is None else device
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]
    model = VelocityField(
        data_dim=2,
        time_emb_dim=config["time_emb_dim"],
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


def sample_flow_matching_ode(
    model: VelocityField,
    n_samples: int = 2000,
    n_steps: int = 250,
    seed: int = 0,
    method: str = "heun",
    base_std: float = 1.0,
    device: torch.device | None = None,
    return_trajectory: bool = False,
    record_every: int = 1,
) -> torch.Tensor | tuple[torch.Tensor, np.ndarray]:
    """
    Integrate the learned Flow Matching ODE from t=0 base noise to t=1 data.

    NOTES.md section 8 convention: t=0 is base, t=1 is data, so this is a
    forward-time integration with positive dt.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if method not in {"euler", "heun"}:
        raise ValueError(f"method must be 'euler' or 'heun', got {method!r}")
    device = default_device() if device is None else device
    torch.manual_seed(seed)

    x = torch.randn(n_samples, 2, device=device) * base_std
    dt = 1.0 / n_steps
    step_fn = heun_step if method == "heun" else euler_step
    trajectory = []
    if return_trajectory:
        trajectory.append(x.detach().cpu().numpy())

    def drift_fn(x_batch: torch.Tensor, t_batch: torch.Tensor) -> torch.Tensor:
        return model(x_batch, torch.clamp(t_batch, 0.0, 1.0))

    with torch.no_grad():
        for step in range(n_steps):
            t = torch.full((n_samples,), step * dt, device=device)
            x = step_fn(x, t, dt, drift_fn)
            if return_trajectory and ((step + 1) % record_every == 0 or step == n_steps - 1):
                trajectory.append(x.detach().cpu().numpy())

    if return_trajectory:
        return x, np.stack(trajectory, axis=0)
    return x
