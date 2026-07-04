"""
Registry mapping distribution names to sampling callables.

Interface for all entries:
    fn(n: int, seed: int = 0) -> np.ndarray  shape (n, 2)

Add a new distribution by inserting one line here — no other code changes needed.
"""

from .distributions import (
    checkerboard,
    eight_gaussians,
    pinwheel,
    two_moons,
    two_spirals,
)

REGISTRY: dict[str, callable] = {
    "eight_gaussians": eight_gaussians,
    "two_moons":       two_moons,
    "checkerboard":    checkerboard,
    "pinwheel":        pinwheel,
    "two_spirals":     two_spirals,
}


def get_distribution(name: str):
    """Return the sampling function for *name*, raising KeyError if unknown."""
    if name not in REGISTRY:
        available = ", ".join(sorted(REGISTRY))
        raise KeyError(f"Unknown distribution {name!r}. Available: {available}")
    return REGISTRY[name]
