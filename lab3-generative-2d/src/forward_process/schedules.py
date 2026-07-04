"""
Shared linear beta-noise-schedule utilities used by VP and sub-VP SDEs.

Reference: references/NOTES.md §1 (VP SDE), used identically in §3 (sub-VP).
"""

import torch


def beta_linear(t: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:
    """beta(t) = beta_min + t (beta_max - beta_min).   NOTES.md §1."""
    return beta_min + t * (beta_max - beta_min)


def beta_integral(t: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:
    """
    int_0^t beta(s) ds = beta_min * t + 1/2 (beta_max - beta_min) t^2.

    Closed form for the linear schedule, used to build alpha(t) in the
    VP/sub-VP transition kernels.   NOTES.md §1, §4.
    """
    return beta_min * t + 0.5 * (beta_max - beta_min) * t ** 2
