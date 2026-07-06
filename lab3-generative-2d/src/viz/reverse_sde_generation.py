"""
Animation 5: reverse-SDE generation from noise to data.

The sampler starts from x(t=1) ~ N(0,I), integrates the reverse-time SDE
back to t approx 0, and draws trajectories for a small particle subset.

CLUTTER FIX (post-hoc review, step 6): drawing the FULL accumulated history
of a stochastic reverse-SDE path (a jagged random walk, unlike the smooth
deterministic PF-ODE path) buried the final cluster structure in line
noise -- confirmed by comparing against pf_ode_generation's clean result
and against sampler_comparison's clean 8-cluster output for the same
checkpoint. Fix: only draw a trailing window (comet-tail) of each path
instead of the full history, and draw final-position markers larger, fully
opaque, and on top (zorder) so the converged structure reads clearly.
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

from src.sampling import load_score_wrapper, sample_reverse_sde
from src.sampling.utils import default_device

OUT_DIR = PROJECT_ROOT / "outputs" / "videos"
KEYFRAME_DIR = PROJECT_ROOT / "outputs" / "sanity_checks"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "eight_gaussians_eps_seed0.pt"))
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--n_particles", type=int, default=60)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--record_every", type=int, default=2)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--axis_limit", type=float, default=6.5)
    p.add_argument("--trail_length", type=int, default=25,
                    help="max recorded points of trailing history drawn per particle (comet-tail)")
    return p.parse_args()


def trail_start(idx: int, trail_length: int) -> int:
    return max(0, idx + 1 - trail_length)


def save_keyframes(trajectory, axis_limit, dist_name, param, colors, trail_length):
    KEYFRAME_DIR.mkdir(parents=True, exist_ok=True)
    key_indices = [0, len(trajectory) // 2, len(trajectory) - 1]
    labels = ["noise", "mid", "data"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), dpi=110)
    for ax, idx, label in zip(axes, key_indices, labels):
        start = trail_start(idx, trail_length)
        for p_idx in range(trajectory.shape[1]):
            path = trajectory[start: idx + 1, p_idx, :]
            ax.plot(path[:, 0], path[:, 1], color=colors[p_idx], alpha=0.35, linewidth=0.6, zorder=2)
        # final positions drawn last/on top, large and fully opaque
        ax.scatter(trajectory[idx, :, 0], trajectory[idx, :, 1], color=colors, s=26,
                   edgecolor="black", linewidth=0.4, zorder=5)
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=12)
        ax.grid(True, linewidth=0.3, alpha=0.4)

    fig.suptitle(f"reverse-SDE generation - {dist_name}/{param}", fontsize=13)
    fig.tight_layout()
    out_path = KEYFRAME_DIR / f"reverse_sde_generation_{dist_name}_{param}_keyframes.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved keyframes to {out_path.relative_to(PROJECT_ROOT)}")


def main():
    args = parse_args()
    device = default_device()
    wrapper, process, config = load_score_wrapper(args.checkpoint, device=device)
    dist_name, param = config["distribution"], config["param"]

    _, trajectory = sample_reverse_sde(
        wrapper,
        process,
        n_samples=args.n_particles,
        n_steps=args.steps,
        seed=args.seed,
        device=device,
        return_trajectory=True,
        record_every=args.record_every,
    )

    colors = plt.cm.hsv(np.linspace(0, 1, args.n_particles, endpoint=False))
    save_keyframes(trajectory, args.axis_limit, dist_name, param, colors, args.trail_length)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
    ax.set_xlim(-args.axis_limit, args.axis_limit)
    ax.set_ylim(-args.axis_limit, args.axis_limit)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    title = ax.set_title("")

    lines = [ax.plot([], [], color=colors[p], alpha=0.35, linewidth=0.6, zorder=2)[0] for p in range(args.n_particles)]
    points = ax.scatter(
        trajectory[0, :, 0],
        trajectory[0, :, 1],
        color=colors,
        s=26,
        edgecolor="black",
        linewidth=0.4,
        zorder=5,
    )

    def update(frame_idx):
        start = trail_start(frame_idx, args.trail_length)
        for p_idx in range(args.n_particles):
            path = trajectory[start: frame_idx + 1, p_idx, :]
            lines[p_idx].set_data(path[:, 0], path[:, 1])
        points.set_offsets(trajectory[frame_idx])
        progress = frame_idx / max(1, len(trajectory) - 1)
        title.set_text(f"Reverse-SDE generation - {dist_name}/{param} - progress={progress:.2f}")
        return lines + [points, title]

    anim = animation.FuncAnimation(fig, update, frames=len(trajectory), blit=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "reverse_sde_generation.mp4"
    anim.save(out_path, writer="ffmpeg", fps=args.fps, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
