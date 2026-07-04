"""
Generic single-step numerical integrators.

These functions are agnostic to what is being integrated -- VP, VE,
Probability Flow ODE, Flow Matching, or anything else. They only consume:
    drift_fn(x, t) -> tensor, same shape as x
    diffusion_fn(t) -> tensor, shape (batch,) or broadcastable to x

No forward-process-specific logic lives here.
"""

import math

import torch


def euler_step(x: torch.Tensor, t: torch.Tensor, dt: float, drift_fn) -> torch.Tensor:
    """
    Explicit Euler (order 1, deterministic):
        x_{n+1} = x_n + f(x_n, t_n) * dt
    """
    return x + drift_fn(x, t) * dt


def heun_step(x: torch.Tensor, t: torch.Tensor, dt: float, drift_fn) -> torch.Tensor:
    """
    Heun's method / improved Euler (order 2, deterministic, predictor-corrector):
        k1 = f(x_n, t_n)
        x_pred = x_n + k1 * dt
        k2 = f(x_pred, t_n + dt)
        x_{n+1} = x_n + (k1 + k2) / 2 * dt
    """
    k1 = drift_fn(x, t)
    x_pred = x + k1 * dt
    k2 = drift_fn(x_pred, t + dt)
    return x + 0.5 * (k1 + k2) * dt


def _match_shape(g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if g.dim() < x.dim():
        return g.view(-1, *([1] * (x.dim() - 1)))
    return g


def euler_maruyama_step(
    x: torch.Tensor, t: torch.Tensor, dt: float, drift_fn, diffusion_fn
) -> torch.Tensor:
    """
    Euler-Maruyama (strong order 0.5, weak order 1, stochastic):
        x_{n+1} = x_n + f(x_n, t_n) * dt + g(t_n) * sqrt(dt) * z,  z ~ N(0, I)
    """
    noise = torch.randn_like(x)
    g = _match_shape(diffusion_fn(t), x)
    return x + drift_fn(x, t) * dt + g * math.sqrt(dt) * noise
