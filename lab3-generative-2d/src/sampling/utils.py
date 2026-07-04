"""Shared helpers for generative samplers."""

from __future__ import annotations

from pathlib import Path

import torch

from src.forward_process import VPSDE
from src.models.denoiser import Denoiser, DenoiserWrapper


def default_device() -> torch.device:
    """Use CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_score_wrapper(checkpoint_path: str | Path, device: torch.device | None = None):
    """
    Load a step-5 denoising checkpoint and return (wrapper, process, config).

    The current eps/v checkpoints are trained against the VP marginal kernel,
    so sampling uses VPSDE as the paired forward process.
    """
    device = default_device() if device is None else device
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]
    model = Denoiser(
        data_dim=2,
        time_emb_dim=config["time_emb_dim"],
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    process = VPSDE()
    wrapper = DenoiserWrapper(model, process, param=config["param"])
    return wrapper, process, config


def match_shape(values: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Reshape a batch scalar tensor so it broadcasts against ref."""
    if values.dim() < ref.dim():
        return values.view(-1, *([1] * (ref.dim() - 1)))
    return values


def reverse_time_to_forward_time(s: torch.Tensor, t_min: float) -> torch.Tensor:
    """
    Convert reverse integration time s in [0,1] to forward diffusion time t.

    s=0 means t=1 (pure noise); s=1 means t=0 (data). We clamp away from
    exactly t=0 because eps/v-to-score formulas divide by the VP std(t).
    """
    return torch.clamp(1.0 - s, min=t_min)


def probability_flow_drift(wrapper, process, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Forward-time Probability Flow ODE drift.

    NOTES.md section 7 / Song et al. probability-flow ODE:
        dx = [f(x,t) - 1/2 g(t)^2 score_t(x)] dt
    """
    f = process.drift(x, t)
    g = match_shape(process.diffusion(t), x)
    score = wrapper.score(x, t)
    return f - 0.5 * (g ** 2) * score


def reverse_sde_forward_drift(wrapper, process, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Forward-time drift of the reverse-time SDE.

    NOTES.md section 7 / Song et al. reverse SDE:
        dx = [f(x,t) - g(t)^2 score_t(x)] dt + g(t) d w_bar

    Sampling integrates this equation backward in t via positive reverse time
    s, so callers negate this drift before passing it to a positive-dt step.
    """
    f = process.drift(x, t)
    g = match_shape(process.diffusion(t), x)
    score = wrapper.score(x, t)
    return f - (g ** 2) * score
