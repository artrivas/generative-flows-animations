"""
Closed-form transition kernels q(x_t | x0) = N(mean, std^2 I).

VP and VE kernels are transcribed directly from references/NOTES.md §4
(Song et al. 2021). The sub-VP kernel is NOT directly quoted from a paper
equation number in NOTES.md — it is derived analytically below from the
general linear-SDE solution and cross-checked numerically in
scripts/validate_forward_process.py. Flagged in NOTES.md §3 for manual
verification against Song et al. 2021, Appendix B.

Derivation of the sub-VP marginal (for reference):
    Linear SDE dx = f(t) x dt + g(t) dW with f(t) = -beta(t)/2 has solution
        x_t = alpha(t) x0 + alpha(t) * int_0^t [g(s)/alpha(s)] dW_s,
        alpha(t) = exp(-1/2 int_0^t beta(s) ds).
    By the Ito isometry, Var[x_t] = alpha(t)^2 int_0^t g(s)^2/alpha(s)^2 ds.
    For sub-VP, g(s)^2 = beta(s)(1 - exp(-2 B(s))), B(s) = int_0^s beta.
    Substituting and integrating (u = exp(B(s))) gives
        Var[x_t] = (1 - alpha(t)^2)^2,
    i.e. std = 1 - alpha(t)^2 (nonnegative since alpha(t) in (0, 1]).
    Note (1-alpha^2)^2 <= 1-alpha^2 for alpha^2 in [0,1], which is the
    "sub" (sub-variance, bounded by VP's variance) property that gives the
    process its name.
"""

import torch

from .schedules import beta_integral


def _match_shape(param: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Reshape a (batch,) tensor to broadcast against ref's trailing dims."""
    return param.view(-1, *([1] * (ref.dim() - 1)))


def vp_transition_kernel(x0: torch.Tensor, t: torch.Tensor, beta_min: float, beta_max: float):
    """
    mean = alpha(t) x0,  std = sqrt(1 - alpha(t)^2)
    alpha(t) = exp(-1/2 int_0^t beta(s) ds).   NOTES.md §4.
    """
    B = beta_integral(t, beta_min, beta_max)
    alpha = torch.exp(-0.5 * B)
    mean = _match_shape(alpha, x0) * x0
    std = torch.sqrt(1 - alpha ** 2)
    return mean, _match_shape(std, x0)


def ve_transition_kernel(x0: torch.Tensor, t: torch.Tensor, sigma_min: float, sigma_max: float):
    """
    mean = x0,  std = sqrt(sigma(t)^2 - sigma_min^2)
    sigma(t) = sigma_min (sigma_max/sigma_min)^t.   NOTES.md §4.
    """
    sigma_t = sigma_min * (sigma_max / sigma_min) ** t
    mean = x0
    std = torch.sqrt(sigma_t ** 2 - sigma_min ** 2)
    return mean, _match_shape(std, x0)


def subvp_transition_kernel(x0: torch.Tensor, t: torch.Tensor, beta_min: float, beta_max: float):
    """
    mean = alpha(t) x0 (same alpha as VP),  std = 1 - alpha(t)^2.

    Derived, not directly quoted from NOTES.md §3 — see module docstring
    for the full derivation. Numerically validated in
    scripts/validate_forward_process.py.
    """
    B = beta_integral(t, beta_min, beta_max)
    alpha = torch.exp(-0.5 * B)
    mean = _match_shape(alpha, x0) * x0
    std = 1 - alpha ** 2
    return mean, _match_shape(std, x0)
