"""
Sanity-check script: scatter-plots n=10000 samples from every registered
distribution and saves them to outputs/sanity_checks/.

All axes use the same fixed range [-5.5, 5.5] so distributions can be
compared side-by-side without perceptual distortion.

Usage:
    python scripts/check_distributions.py
"""

import sys
from pathlib import Path

# Allow running from any cwd by adding the project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.registry import REGISTRY

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_SAMPLES  = 10_000
SEED       = 42
AXIS_LIMIT = 5.5           # shared for all plots
PT_SIZE    = 0.8           # scatter point size
ALPHA      = 0.35
OUT_DIR    = PROJECT_ROOT / "outputs" / "sanity_checks"

COLORS = [
    "#4C72B0", "#DD8452", "#55A868",
    "#C44E52", "#8172B3", "#937860",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_scatter(ax, samples: np.ndarray, title: str, color: str) -> None:
    ax.scatter(
        samples[:, 0], samples[:, 1],
        s=PT_SIZE, alpha=ALPHA, color=color, rasterized=True,
        linewidths=0,
    )
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xlim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_ylim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.3, alpha=0.4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    names = list(REGISTRY.keys())
    n_dist = len(names)

    # ---- individual plots --------------------------------------------------
    for name, color in zip(names, COLORS):
        fn = REGISTRY[name]
        samples = fn(N_SAMPLES, seed=SEED)

        assert samples.shape == (N_SAMPLES, 2), (
            f"{name}: expected ({N_SAMPLES}, 2), got {samples.shape}"
        )
        assert np.isfinite(samples).all(), f"{name}: contains non-finite values"

        fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
        make_scatter(ax, samples, name, color)
        fig.tight_layout()
        path = OUT_DIR / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path.relative_to(PROJECT_ROOT)}")

    # ---- overview grid -----------------------------------------------------
    ncols = 3
    nrows = -(-n_dist // ncols)   # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), dpi=120)
    axes_flat = axes.flat

    for ax, name, color in zip(axes_flat, names, COLORS):
        samples = REGISTRY[name](N_SAMPLES, seed=SEED)
        make_scatter(ax, samples, name, color)

    # hide unused subplots
    for ax in list(axes_flat)[n_dist:]:
        ax.set_visible(False)

    fig.suptitle("Distribution Sanity Check  (n=10 000, fixed axes ±5.5)", fontsize=13, y=1.01)
    fig.tight_layout()
    overview_path = OUT_DIR / "overview.png"
    fig.savefig(overview_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {overview_path.relative_to(PROJECT_ROOT)}")

    # ---- quick stats -------------------------------------------------------
    print("\n--- Stats (mean ± std per axis) ---")
    for name in names:
        s = REGISTRY[name](N_SAMPLES, seed=SEED)
        print(
            f"  {name:<18}  "
            f"x: {s[:,0].mean():+.3f} ± {s[:,0].std():.3f}  "
            f"y: {s[:,1].mean():+.3f} ± {s[:,1].std():.3f}  "
            f"|max|: {np.abs(s).max():.3f}"
        )


if __name__ == "__main__":
    main()
