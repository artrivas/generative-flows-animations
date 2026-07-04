"""
Animation 3: forward process trajectories for a small particle subset.

Shows how individual particles move from x0 to the near-prior final state
of the forward process, with trajectories drawn as GROWING LINES (the
path history stays visible), not just the current position.

Uses the same "fixed per-particle noise z" trick as forward_comparison.py:
x_t(z) = mean(t) + std(t) * z, z ~ N(0, I) fixed per particle. This
reproduces the exact VP marginal at every t while producing smooth,
continuous-looking paths (independent resampling per frame would not
trace a coherent trajectory -- it would look like teleporting dots).
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
from src.forward_process import VPSDE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DISTRIBUTION = "eight_gaussians"
N_PARTICLES = 40
N_FRAMES = 150
FPS = 24
SEED = 3

OUT_DIR = PROJECT_ROOT / "outputs" / "videos"
KEYFRAME_DIR = PROJECT_ROOT / "outputs" / "sanity_checks"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--distribution", default=DISTRIBUTION)
    p.add_argument("--n_particles", type=int, default=N_PARTICLES)
    p.add_argument("--n_frames", type=int, default=N_FRAMES)
    p.add_argument("--fps", type=int, default=FPS)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def build_trajectories():
    torch.manual_seed(SEED)
    sample_fn = get_distribution(DISTRIBUTION)
    x0_np = sample_fn(N_PARTICLES, seed=SEED)
    x0 = torch.from_numpy(x0_np).float()

    z = torch.randn(N_PARTICLES, 2)  # fixed per-particle noise, reused across all frames
    process = VPSDE()

    t_values = np.linspace(0.0, 1.0, N_FRAMES)
    positions = np.zeros((N_FRAMES, N_PARTICLES, 2))
    for i, t_val in enumerate(t_values):
        t = torch.full((N_PARTICLES,), float(t_val))
        mean, std = process.marginal_prob(x0, t)
        positions[i] = (mean + std * z).numpy()

    limit_raw = np.percentile(np.abs(positions.reshape(-1)), 99.5) * 1.15
    axis_limit = float(np.ceil(limit_raw / 2.0) * 2.0)

    return positions, t_values, axis_limit


def save_keyframes(positions, t_values, axis_limit, colors):
    KEYFRAME_DIR.mkdir(parents=True, exist_ok=True)
    key_indices = [0, len(t_values) // 2, len(t_values) - 1]

    fig, axes = plt.subplots(1, len(key_indices), figsize=(16, 5.5), dpi=110)
    for ax, idx in zip(axes, key_indices):
        for p in range(N_PARTICLES):
            path = positions[: idx + 1, p, :]
            ax.plot(path[:, 0], path[:, 1], color=colors[p], alpha=0.55, linewidth=0.9)
            ax.scatter(path[-1, 0], path[-1, 1], color=colors[p], s=18, zorder=5, edgecolor="black", linewidth=0.3)
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)
        ax.set_aspect("equal")
        ax.set_title(f"t = {t_values[idx]:.2f}", fontsize=12)
        ax.grid(True, linewidth=0.3, alpha=0.4)

    fig.suptitle("forward_trajectories — key frames (VP)", fontsize=14)
    fig.tight_layout()
    path_out = KEYFRAME_DIR / "forward_trajectories_keyframes.png"
    fig.savefig(path_out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved keyframes to {path_out.relative_to(PROJECT_ROOT)}")


def main():
    args = parse_args()
    global DISTRIBUTION, N_PARTICLES, N_FRAMES, FPS, SEED
    DISTRIBUTION = args.distribution
    N_PARTICLES = args.n_particles
    N_FRAMES = args.n_frames
    FPS = args.fps
    SEED = args.seed

    positions, t_values, axis_limit = build_trajectories()
    print(f"axis_limit = +/-{axis_limit}")
    colors = plt.cm.hsv(np.linspace(0, 1, N_PARTICLES, endpoint=False))

    save_keyframes(positions, t_values, axis_limit, colors)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    title = ax.set_title("")

    lines = [ax.plot([], [], color=colors[p], alpha=0.55, linewidth=0.9)[0] for p in range(N_PARTICLES)]
    points = ax.scatter(positions[0, :, 0], positions[0, :, 1], color=colors, s=18, zorder=5,
                         edgecolor="black", linewidth=0.3)

    def update(frame_idx):
        for p in range(N_PARTICLES):
            path = positions[: frame_idx + 1, p, :]
            lines[p].set_data(path[:, 0], path[:, 1])
        points.set_offsets(positions[frame_idx])
        title.set_text(f"Forward process trajectories (VP) — t = {t_values[frame_idx]:.2f}")
        return lines + [points, title]

    anim = animation.FuncAnimation(fig, update, frames=N_FRAMES, blit=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "forward_trajectories.mp4"
    anim.save(out_path, writer="ffmpeg", fps=FPS, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
