"""
Structure-sensitive metrics for the pinwheel distribution.

Energy distance (used everywhere else) is dominated by radial/global spread
and is documented (docs/technical.md §9.4) as blind to pinwheel's specific
failure mode: an isotropic blob can score a LOW (good-looking) energy
distance against the real 5-arm data as long as its radial profile roughly
matches, even though all angular/spiral structure is gone.

Both metrics below look at angle, not just distance-from-real-points:

- arm_assignment_accuracy: pinwheel's exact generative process (see
  src/data/distributions.py::pinwheel) says a point belongs to arm k if
  theta ~= k*2*pi/n_arms + t*pi for some t in [0, 1], with r ~= 3*t.
  IMPORTANT: t must be estimated from theta, not from r. angular_std=0.04
  rad is tight, but radial_std=0.25 propagated through r/3.0*pi is a much
  larger ~0.26 rad of apparent angular slop -- inverting r -> t and then
  checking the angular residual (the naive approach) is dominated by that
  propagated radial noise and gives real data itself only ~0.3 accuracy.
  Instead: for each candidate arm k, invert theta -> t_hat = wrap(theta -
  k*2*pi/n_arms) / pi (precise, since angular noise is small), then check
  whether the observed r is consistent with r_scale * t_hat within a
  radius tolerance (radial_std-based). This correctly gives real data
  accuracy near 1.0, and a collapsed isotropic blob a low accuracy near
  its chance level (estimated empirically by estimate_chance_level below,
  since deriving it in closed form for this asymmetric r-given-theta test
  is not a simple formula).

- angular_histogram_distance: marginal histogram of theta = atan2(y, x),
  restricted to a near-origin radial band (r in [0.15, 0.8] by default),
  compared between generated and real samples via total-variation (L1)
  distance. Pooling across ALL radii was tried first and rejected: each
  arm curls a full pi radians over its length, so the marginal angle
  histogram pooled over all r is already fairly smeared/broad for real
  data too, making real-vs-real sampling noise (~0.09 at n=5000) almost as
  large as real-vs-isotropic-blob (~0.15) -- a weak, unreliable signal.
  Restricting to a near-origin band, where the 5 arms are still angularly
  well-separated (little curl yet), roughly triples the separation
  (real-vs-real ~0.16 vs blob-vs-real ~0.47 at the same sample size),
  confirmed empirically before adopting these defaults.
"""

import numpy as np

N_ARMS = 5
ARM_R_SCALE = 3.0  # matches src/data/distributions.py::pinwheel's r = 3.0 * t
RADIAL_STD = 0.25  # matches src/data/distributions.py::pinwheel's radial noise
ANGULAR_STD = 0.04  # matches src/data/distributions.py::pinwheel's angular noise
# 1-sigma band on the radius-given-arm residual. Wider bands (e.g. 3-sigma) let
# an isotropic blob "accidentally" satisfy some arm's predicted radius too
# often to be useful -- pinwheel's 5 arms each curl a full pi radians while
# their base angles are only 2*pi/5 apart, so a generous tolerance combined
# with 5 candidate arms to check against a single point is a multiple-
# comparisons problem that inflates the chance-level pass rate a lot faster
# than it inflates real data's pass rate. 1-sigma keeps the chance level and
# real-data level clearly separated (see estimate_chance_level / the
# real-data calibration accuracy printed by scripts/evaluate_pinwheel_structure.py).
DEFAULT_RADIAL_TOL = RADIAL_STD


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def arm_assignment_accuracy(
    samples: np.ndarray,
    n_arms: int = N_ARMS,
    r_scale: float = ARM_R_SCALE,
    radial_tol: float = DEFAULT_RADIAL_TOL,
):
    """
    Returns (accuracy, best_residual_per_point) where best_residual is a
    radius residual (same units as the data), so callers can also inspect
    per-point pass/fail via best_residual < radial_tol.
    """
    r = np.linalg.norm(samples, axis=1)
    theta = np.arctan2(samples[:, 1], samples[:, 0])

    best_resid = np.full(len(samples), np.inf)
    for k in range(n_arms):
        delta = _wrap_to_pi(theta - k * 2 * np.pi / n_arms)
        t_hat = delta / np.pi
        valid = (t_hat > -0.05) & (t_hat < 1.15)
        r_pred = np.clip(t_hat, 0.0, 1.15) * r_scale
        resid = np.where(valid, np.abs(r - r_pred), np.inf)
        best_resid = np.minimum(best_resid, resid)

    accuracy = float(np.mean(best_resid < radial_tol))
    return accuracy, best_resid


def estimate_chance_level(n: int = 20000, r_max: float = 4.5, seed: int = 0, **kwargs) -> float:
    """
    Empirical arm_assignment_accuracy for an isotropic (uniform-angle) blob
    with the same rough radial extent as pinwheel data -- i.e. what a total
    collapse-to-blob failure scores "for free", for calibration.
    """
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.0, r_max, size=n)
    theta = rng.uniform(-np.pi, np.pi, size=n)
    samples = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    accuracy, _ = arm_assignment_accuracy(samples, **kwargs)
    return accuracy


def angular_histogram(
    samples: np.ndarray, n_bins: int = 36, r_min: float = 0.15, r_max: float = 0.8
) -> np.ndarray:
    """Density histogram of theta = atan2(y, x) within the [r_min, r_max) band."""
    r = np.linalg.norm(samples, axis=1)
    theta = np.arctan2(samples[:, 1], samples[:, 0])
    theta = theta[(r >= r_min) & (r < r_max)]
    hist, _ = np.histogram(theta, bins=n_bins, range=(-np.pi, np.pi), density=True)
    return hist


def angular_histogram_distance(
    gen: np.ndarray, real: np.ndarray, n_bins: int = 36, r_min: float = 0.15, r_max: float = 0.8
) -> float:
    """Total-variation (L1) distance between the two angular histograms, in [0, 2]."""
    bin_width = 2 * np.pi / n_bins
    p = angular_histogram(gen, n_bins=n_bins, r_min=r_min, r_max=r_max) * bin_width
    q = angular_histogram(real, n_bins=n_bins, r_min=r_min, r_max=r_max) * bin_width
    return float(np.abs(p - q).sum())
