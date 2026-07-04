"""
Animation 2: density evolution of the forward process, per distribution.

Shows how the empirical density of a large particle cloud smooths out from
the original shape (multimodal / curved support) towards the prior
N(0, I) as t sweeps from 0 to 1, under the VP SDE. VP is used as the
representative "standard" forward process for this visualization (it's
what VE and sub-VP are usually benchmarked against); swap PROCESS_FACTORY
below to visualize a different one.

Particles reuse a FIXED per-particle noise vector z (same trick as
forward_comparison.py) so density estimates evolve continuously across
frames instead of flickering from independent resampling at every frame --
that flicker would look exactly like the "spurious peaks" this animation
must NOT show.

Density is estimated via a 2D histogram (fast for N=30000) followed by a
light Gaussian smoothing (scipy.ndimage.gaussian_filter) to remove binning
artifacts. Each frame's color scale is normalized independently (vmin=0,
vmax=99.5th percentile of THAT frame's density) so the *shape* stays
legible even as the true density flattens out approaching the prior --
color intensity is therefore not directly comparable across frames, only
the shape is.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import imageio_ffmpeg

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from src.data.registry import get_distribution
from src.forward_process import VPSDE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DISTRIBUTIONS = ["eight_gaussians", "two_moons"]  # multimodal + curved support
N_PARTICLES = 30_000
N_FRAMES = 120
FPS = 24
SEED = 11
BINS = 70
SMOOTH_SIGMA = 1.0
AXIS_LIMIT = 7.0  # fixed window; VP stays bounded (std<=1, x0 spread ~2-2.5)

OUT_DIR = PROJECT_ROOT / "outputs" / "videos"
KEYFRAME_DIR = PROJECT_ROOT / "outputs" / "sanity_checks"


def compute_density(x: np.ndarray, bins: int, limit: float) -> np.ndarray:
    hist, _, _ = np.histogram2d(
        x[:, 0], x[:, 1], bins=bins, range=[[-limit, limit], [-limit, limit]]
    )
    hist = gaussian_filter(hist, sigma=SMOOTH_SIGMA)
    return hist.T  # imshow expects [row=y, col=x]


def build_densities(dist_name: str):
    proc = VPSDE()
    sample_fn = get_distribution(dist_name)
    x0_np = sample_fn(N_PARTICLES, seed=SEED)
    x0 = torch.from_numpy(x0_np).float()

    torch.manual_seed(SEED)
    z = torch.randn(N_PARTICLES, 2)

    t_values = np.linspace(0.0, 1.0, N_FRAMES)
    densities = []
    for t_val in t_values:
        t = torch.full((N_PARTICLES,), float(t_val))
        mean, std = proc.marginal_prob(x0, t)
        x_t = (mean + std * z).numpy()
        densities.append(compute_density(x_t, BINS, AXIS_LIMIT))

    return densities, t_values


def save_keyframes(dist_name: str, densities, t_values):
    KEYFRAME_DIR.mkdir(parents=True, exist_ok=True)
    key_indices = [0, len(t_values) // 2, len(t_values) - 1]

    fig, axes = plt.subplots(1, len(key_indices), figsize=(15, 5.2), dpi=110)
    for ax, idx in zip(axes, key_indices):
        d = densities[idx]
        vmax = max(np.percentile(d, 99.5), 1e-6)
        ax.imshow(
            d, extent=[-AXIS_LIMIT, AXIS_LIMIT, -AXIS_LIMIT, AXIS_LIMIT],
            origin="lower", cmap="magma", vmin=0, vmax=vmax,
        )
        ax.set_title(f"t = {t_values[idx]:.2f}", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"density_evolution — {dist_name} (VP)", fontsize=14)
    fig.tight_layout()
    path = KEYFRAME_DIR / f"density_evolution_{dist_name}_keyframes.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved keyframes to {path.relative_to(PROJECT_ROOT)}")


def render(dist_name: str):
    densities, t_values = build_densities(dist_name)
    save_keyframes(dist_name, densities, t_values)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    vmax0 = max(np.percentile(densities[0], 99.5), 1e-6)
    im = ax.imshow(
        densities[0], extent=[-AXIS_LIMIT, AXIS_LIMIT, -AXIS_LIMIT, AXIS_LIMIT],
        origin="lower", cmap="magma", vmin=0, vmax=vmax0,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title(f"{dist_name} — VP density evolution, t=0.00", fontsize=12)

    def update(frame_idx):
        d = densities[frame_idx]
        im.set_data(d)
        im.set_clim(0, max(np.percentile(d, 99.5), 1e-6))
        title.set_text(f"{dist_name} — VP density evolution, t={t_values[frame_idx]:.2f}")
        return [im, title]

    anim = animation.FuncAnimation(fig, update, frames=N_FRAMES, blit=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"density_evolution_{dist_name}.mp4"
    anim.save(out_path, writer="ffmpeg", fps=FPS, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path.relative_to(PROJECT_ROOT)}")


def main():
    for dist_name in DISTRIBUTIONS:
        render(dist_name)


if __name__ == "__main__":
    main()
