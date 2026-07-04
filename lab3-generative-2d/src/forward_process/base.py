"""
Common interface for forward diffusion processes.

    dx = drift(x, t) dt + diffusion(t) dW
"""

from abc import ABC, abstractmethod
from typing import Tuple

import torch


class ForwardProcess(ABC):
    """Abstract forward SDE: dx = f(x,t) dt + g(t) dW."""

    @abstractmethod
    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """f(x, t): drift term of the forward SDE. x: (batch, dim), t: (batch,)."""

    @abstractmethod
    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """g(t): diffusion coefficient (scalar per t, shared across dims). t: (batch,)."""

    @abstractmethod
    def marginal_prob(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Closed-form transition kernel q(x_t | x0) = N(mean, std^2 I).

        Returns:
            mean: same shape as x0, (batch, dim)
            std:  broadcastable to x0, (batch, 1)
        """
