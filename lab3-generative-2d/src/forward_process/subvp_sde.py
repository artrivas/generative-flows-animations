"""
Sub-Variance Preserving (sub-VP) SDE.

Reference: references/NOTES.md §3 — Song et al. 2021, Appendix B
(drift/diffusion). The marginal_prob kernel is derived, not directly
quoted — see the derivation in src/forward_process/kernels.py and the
flag in NOTES.md §3.
"""

import torch

from .base import ForwardProcess
from .kernels import subvp_transition_kernel
from .schedules import beta_integral, beta_linear


class SubVPSDE(ForwardProcess):
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """f(x, t) = -1/2 beta(t) x (identical to VP).   NOTES.md §3."""
        beta_t = beta_linear(t, self.beta_min, self.beta_max)
        beta_t = beta_t.view(-1, *([1] * (x.dim() - 1)))
        return -0.5 * beta_t * x

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """
        g(t) = sqrt(beta(t) (1 - exp(-2 int_0^t beta(s) ds))).   NOTES.md §3.
        """
        B = beta_integral(t, self.beta_min, self.beta_max)
        beta_t = beta_linear(t, self.beta_min, self.beta_max)
        return torch.sqrt(beta_t * (1 - torch.exp(-2 * B)))

    def marginal_prob(self, x0: torch.Tensor, t: torch.Tensor):
        """
        q(x_t | x0) = N(alpha(t) x0, (1 - alpha(t)^2)^2 I),
        i.e. std = 1 - alpha(t)^2, alpha(t) = exp(-1/2 int_0^t beta(s) ds).

        Derived analytically from the linear-SDE solution (Ito isometry) —
        see src/forward_process/kernels.py docstring. Flagged in NOTES.md
        §3 for manual verification against Song et al. 2021, Appendix B.
        """
        return subvp_transition_kernel(x0, t, self.beta_min, self.beta_max)
