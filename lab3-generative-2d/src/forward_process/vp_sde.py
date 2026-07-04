"""
Variance Preserving (VP) SDE.

Reference: references/NOTES.md §1 (drift/diffusion) and §4 (closed-form
kernel) — Song et al. 2021, "Score-Based Generative Modeling through SDEs".
"""

import torch

from .base import ForwardProcess
from .kernels import vp_transition_kernel
from .schedules import beta_linear


class VPSDE(ForwardProcess):
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """f(x, t) = -1/2 beta(t) x.   NOTES.md §1."""
        beta_t = beta_linear(t, self.beta_min, self.beta_max)
        beta_t = beta_t.view(-1, *([1] * (x.dim() - 1)))
        return -0.5 * beta_t * x

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """g(t) = sqrt(beta(t)).   NOTES.md §1."""
        beta_t = beta_linear(t, self.beta_min, self.beta_max)
        return torch.sqrt(beta_t)

    def marginal_prob(self, x0: torch.Tensor, t: torch.Tensor):
        """
        q(x_t | x0) = N(alpha(t) x0, (1 - alpha(t)^2) I),
        alpha(t) = exp(-1/2 int_0^t beta(s) ds).   NOTES.md §4.
        """
        return vp_transition_kernel(x0, t, self.beta_min, self.beta_max)
