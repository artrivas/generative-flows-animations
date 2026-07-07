"""
Animation 1: side-by-side comparison of VP / VE / sub-VP forward processes
on the same initial samples.

Particles are advanced using the validated closed-form marginal kernel
(marginal_prob), NOT by integrating the SDE step by step. To get smooth,
continuous-looking per-particle trajectories (rather than independent
resamples at every frame, which would look like flicker), each particle
gets a FIXED standard-normal vector z once; its position at time t is the
deterministic function

    x_t(z) = mean(t) + std(t) * z

Since z ~ N(0, I), x_t(z) has exactly the (mean, std) that marginal_prob
predicts at every t -- this is a valid way to visualize the marginal
evolution, just not a literal single Wiener-process sample path.

REAL PRODUCTION HYPERPARAMETERS: VP/sub-VP use beta_max=20 and VE uses
sigma_max=50, matching the validated defaults used everywhere else in the
codebase (forward_process validation, training, sampling). An earlier
version of this script secretly weakened these (beta_max=4, sigma_max=4)
to make the plot "look nicer" -- that silently misrepresented the actual
configured processes and has been removed. VE's real spread (~50x that of
VP/sub-VP, whose std is bounded by 1) is handled with a symlog DISPLAY
axis (see AXIS_LINTHRESH below) so both scales are readable in the same
shared coordinate frame -- this rescales only how the plot is drawn, not
the underlying process.
"""

import argparse
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

from src.data.registry import get_distribution
from src.forward_process import VESDE, VPSDE, SubVPSDE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DISTRIBUTION = "eight_gaussians"
N_PARTICLES = 1500
N_FRAMES = 120
FPS = 24
SEED = 7
VE_SIGMA_MAX = 50.0  # real production default -- see module docstring
BETA_MAX_COMPARE = 20.0  # real production default -- see module docstring
AXIS_LINTHRESH = 2.0  # symlog linear region half-width (display-only rescaling)
POINT_SIZE = 5
ALPHA = 0.5
COLOR = "#4C72B0"

OUT_DIR = PROJECT_ROOT / "outputs" / "videos"
KEYFRAME_DIR = PROJECT_ROOT / "outputs" / "sanity_checks"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--distribution", default=DISTRIBUTION)
    p.add_argument("--n_particles", type=int, default=N_PARTICLES)
    p.add_argument("--n_frames", type=int, default=N_FRAMES)
    p.add_argument("--fps", type=int, default=FPS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--ve_sigma_max", type=float, default=VE_SIGMA_MAX)
    p.add_argument("--beta_max_compare", type=float, default=BETA_MAX_COMPARE)
    return p.parse_args()


def build_frames():
    """Precompute x_t for every (process, frame). Returns (all_frames, t_values, axis_limit)."""
    torch.manual_seed(SEED)

    sample_fn = get_distribution(DISTRIBUTION)
    x0_np = sample_fn(N_PARTICLES, seed=SEED)
    x0 = torch.from_numpy(x0_np).float()

    z = torch.randn(N_PARTICLES, 2)  # fixed per-particle noise, reused across t and processes

    processes = {
        "VP": VPSDE(beta_max=BETA_MAX_COMPARE),
        "VE": VESDE(sigma_max=VE_SIGMA_MAX),
        "sub-VP": SubVPSDE(beta_max=BETA_MAX_COMPARE),
    }

    t_values = np.linspace(0.0, 1.0, N_FRAMES)

    all_frames = {name: [] for name in processes}
    for name, proc in processes.items():
        for t_val in t_values:
            t = torch.full((N_PARTICLES,), float(t_val))
            mean, std = proc.marginal_prob(x0, t)
            x_t = mean + std * z
            all_frames[name].append(x_t.numpy())

    all_coords = np.concatenate(
        [np.stack(all_frames[name]).reshape(-1) for name in processes]
    )
    limit_raw = np.percentile(np.abs(all_coords), 99.5) * 1.15
    axis_limit = float(np.ceil(limit_raw / 5.0) * 5.0)

    return all_frames, t_values, axis_limit, list(processes.keys())


def apply_symlog(ax, axis_limit):
    ax.set_xscale("symlog", linthresh=AXIS_LINTHRESH)
    ax.set_yscale("symlog", linthresh=AXIS_LINTHRESH)
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)


def make_figure(all_frames, axis_limit, names):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), dpi=120)
    scatters = {}
    for ax, name in zip(axes, names):
        apply_symlog(ax, axis_limit)
        ax.set_title(name, fontsize=13)
        ax.grid(True, linewidth=0.3, alpha=0.4)
        sc = ax.scatter([], [], s=POINT_SIZE, alpha=ALPHA, color=COLOR, linewidths=0)
        scatters[name] = sc
    return fig, axes, scatters


def save_keyframes(all_frames, t_values, axis_limit, names):
    """Save static PNGs at t=0, 0.5, 1.0 for visual inspection."""
    KEYFRAME_DIR.mkdir(parents=True, exist_ok=True)
    key_indices = [0, len(t_values) // 2, len(t_values) - 1]

    fig, axes = plt.subplots(len(key_indices), len(names), figsize=(15, 15), dpi=110)
    for row, idx in enumerate(key_indices):
        for col, name in enumerate(names):
            ax = axes[row, col]
            pts = all_frames[name][idx]
            ax.scatter(pts[:, 0], pts[:, 1], s=POINT_SIZE, alpha=ALPHA, color=COLOR, linewidths=0)
            apply_symlog(ax, axis_limit)
            ax.set_title(f"{name}  (t={t_values[idx]:.2f})", fontsize=11)
            ax.grid(True, linewidth=0.3, alpha=0.4)

    fig.suptitle("forward_comparison — key frames", fontsize=14, y=1.0)
    fig.tight_layout()
    path = KEYFRAME_DIR / "forward_comparison_keyframes.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved keyframes to {path.relative_to(PROJECT_ROOT)}")


def main():
    args = parse_args()
    global DISTRIBUTION, N_PARTICLES, N_FRAMES, FPS, SEED, VE_SIGMA_MAX, BETA_MAX_COMPARE
    DISTRIBUTION = args.distribution
    N_PARTICLES = args.n_particles
    N_FRAMES = args.n_frames
    FPS = args.fps
    SEED = args.seed
    VE_SIGMA_MAX = args.ve_sigma_max
    BETA_MAX_COMPARE = args.beta_max_compare

    all_frames, t_values, axis_limit, names = build_frames()
    print(f"axis_limit = +/-{axis_limit}")

    save_keyframes(all_frames, t_values, axis_limit, names)

    fig, axes, scatters = make_figure(all_frames, axis_limit, names)
    time_text = fig.suptitle("", fontsize=13)

    def update(frame_idx):
        t_val = t_values[frame_idx]
        for name in scatters:
            scatters[name].set_offsets(all_frames[name][frame_idx])
        time_text.set_text(f"Forward process comparison  —  t = {t_val:.2f}")
        return list(scatters.values()) + [time_text]

    anim = animation.FuncAnimation(fig, update, frames=N_FRAMES, blit=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "forward_comparison.mp4"
    anim.save(out_path, writer="ffmpeg", fps=FPS, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
