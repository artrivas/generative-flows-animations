"""
Variance Exploding (VE) SDE.

Reference: references/NOTES.md §2 (drift/diffusion) and §4 (closed-form
kernel) — Song et al. 2021, "Score-Based Generative Modeling through SDEs".
"""

import math

import torch

from .base import ForwardProcess
from .kernels import ve_transition_kernel


class VESDE(ForwardProcess):
    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self._log_ratio = math.log(sigma_max / sigma_min)

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """f(x, t) = 0 (no drift).   NOTES.md §2."""
        return torch.zeros_like(x)

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """g(t) = sigma(t) sqrt(2 log(sigma_max/sigma_min)).   NOTES.md §2."""
        sigma_t = self._sigma(t)
        return sigma_t * math.sqrt(2 * self._log_ratio)

    def marginal_prob(self, x0: torch.Tensor, t: torch.Tensor):
        """
        q(x_t | x0) = N(x0, (sigma(t)^2 - sigma_min^2) I),
        sigma(t) = sigma_min (sigma_max/sigma_min)^t.   NOTES.md §4.
        """
        return ve_transition_kernel(x0, t, self.sigma_min, self.sigma_max)
