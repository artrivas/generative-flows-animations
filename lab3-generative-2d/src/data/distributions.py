"""
Synthetic 2D distributions for generative modeling experiments.

All distributions return np.ndarray of shape (n, 2) with comparable spatial
scale: most mass lives inside [-5, 5]^2.
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Multimodal — Eight Gaussians
# ---------------------------------------------------------------------------

def eight_gaussians(n: int, seed: int = 0) -> np.ndarray:
    """8 isotropic Gaussian modes arranged uniformly on a circle of radius 3."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    centers = np.stack([np.cos(angles), np.sin(angles)], axis=1) * 3.0

    idx = rng.integers(0, 8, size=n)
    noise = rng.normal(0.0, 0.4, size=(n, 2))
    return centers[idx] + noise


# ---------------------------------------------------------------------------
# 2. Curved support — Two Moons
# ---------------------------------------------------------------------------

def two_moons(n: int, seed: int = 0) -> np.ndarray:
    """
    Two crescent shapes with clear separation.

    Moon 1: upper arc, angles in [0, pi], y ∈ [0, 1] before scaling.
    Moon 2: lower arc, angles in [pi, 2pi] shifted to (1, -0.5),
            y ∈ [-1.5, -0.5] before scaling → gap ≥ 0.5 between moons.
    Scaled by 2.5 → gap ≥ 1.25 >> noise (σ=0.125).
    """
    rng = np.random.default_rng(seed)
    n_half = n // 2
    n_rest = n - n_half

    t1 = rng.uniform(0, np.pi, n_half)
    moon1 = np.column_stack([np.cos(t1), np.sin(t1)])

    t2 = rng.uniform(np.pi, 2 * np.pi, n_rest)
    moon2 = np.column_stack([np.cos(t2) + 1.0, np.sin(t2) - 0.5])

    samples = np.concatenate([moon1, moon2], axis=0)
    samples += rng.normal(0.0, 0.05, samples.shape)
    return samples * 2.5


# ---------------------------------------------------------------------------
# 3. Disconnected regions — Checkerboard
# ---------------------------------------------------------------------------

def checkerboard(n: int, seed: int = 0) -> np.ndarray:
    """
    Uniform samples in the active cells of a 4×4 checkerboard on [-4, 4]^2.

    Cell (ix, iy) is active iff (ix + iy) % 2 == 0 (8 of 16 cells).
    No noise added — clean discontinuous boundaries.
    """
    rng = np.random.default_rng(seed)
    collected = []
    needed = n
    batch = max(n * 4, 4096)

    while needed > 0:
        xy = rng.uniform(-4.0, 4.0, size=(batch, 2))
        ix = np.floor((xy[:, 0] + 4.0) / 2.0).astype(int).clip(0, 3)
        iy = np.floor((xy[:, 1] + 4.0) / 2.0).astype(int).clip(0, 3)
        mask = (ix + iy) % 2 == 0
        kept = xy[mask]
        collected.append(kept[:needed])
        needed -= len(kept[:needed])

    return np.concatenate(collected, axis=0)[:n]


# ---------------------------------------------------------------------------
# 4. Complex geometry — Pinwheel
# ---------------------------------------------------------------------------

def pinwheel(n: int, seed: int = 0) -> np.ndarray:
    """
    5 spiral arms radiating from the origin.

    Each arm: radial extent ~[0, 3], curls by π radians over its length.
    Radial noise σ=0.25, angular noise σ=0.04 rad.
    """
    rng = np.random.default_rng(seed)
    n_arms = 5
    radial_std = 0.25
    angular_std = 0.04

    sizes = np.full(n_arms, n // n_arms, dtype=int)
    sizes[: n % n_arms] += 1  # distribute remainder

    parts = []
    for k, m in enumerate(sizes):
        t = rng.uniform(0.05, 1.0, size=m)          # avoid exact center
        r = 3.0 * t + rng.normal(0.0, radial_std, size=m)
        theta = (
            k * 2 * np.pi / n_arms              # arm base angle
            + t * np.pi                         # curl: 0 → π over arm length
            + rng.normal(0.0, angular_std, size=m)
        )
        parts.append(np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1))

    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# 5. Extra — Two Spirals (bonus: continuously curved, two connected components)
# ---------------------------------------------------------------------------

def two_spirals(n: int, seed: int = 0) -> np.ndarray:
    """
    Two interleaved Archimedean spirals making 1.5 full turns each.

    Useful contrast to Two Moons: connected components that interlock.
    """
    rng = np.random.default_rng(seed)
    n_half = n // 2
    n_rest = n - n_half

    def arm(m: int, phase: float) -> np.ndarray:
        t = rng.uniform(0.25, 1.0, size=m)          # skip tight center
        r = 3.5 * t
        theta = t * 3.0 * np.pi + phase + rng.normal(0.0, 0.05, size=m)
        r = r + rng.normal(0.0, 0.12, size=m)
        return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

    return np.concatenate([arm(n_half, 0.0), arm(n_rest, np.pi)], axis=0)
