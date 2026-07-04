"""
Animation 7: Flow Matching transport from base noise to data.

Shows particles, accumulated trajectories, and the learned velocity field
at the current time. A static side-by-side path comparison against the
reverse-SDE sampler is also saved when a compatible score checkpoint exists.
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

from src.sampling import (
    load_score_wrapper,
    load_velocity_model,
    sample_flow_matching_ode,
    sample_reverse_sde,
)
from src.sampling.utils import default_device

OUT_DIR = PROJECT_ROOT / "outputs" / "videos"
KEYFRAME_DIR = PROJECT_ROOT / "outputs" / "sanity_checks"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "eight_gaussians_flow_matching_seed0.pt"))
    p.add_argument("--score_checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "eight_gaussians_eps_seed0.pt"))
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--n_particles", type=int, default=60)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--method", choices=["euler", "heun"], default="heun")
    p.add_argument("--record_every", type=int, default=2)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--grid_size", type=int, default=18)
    p.add_argument("--grid_limit", type=float, default=6.5)
    p.add_argument("--axis_limit", type=float, default=6.5)
    return p.parse_args()


def velocity_grid(model, device, grid_size, grid_limit, t_values):
    xs = np.linspace(-grid_limit, grid_limit, grid_size)
    ys = np.linspace(-grid_limit, grid_limit, grid_size)
    XX, YY = np.meshgrid(xs, ys)
    grid = torch.from_numpy(np.stack([XX.ravel(), YY.ravel()], axis=1)).float().to(device)

    all_u, all_v, all_mag = [], [], []
    with torch.no_grad():
        for t_val in t_values:
            t = torch.full((grid.shape[0],), float(t_val), device=device)
            velocity = model(grid, t).detach().cpu().numpy()
            U = velocity[:, 0].reshape(XX.shape)
            V = velocity[:, 1].reshape(XX.shape)
            mag = np.sqrt(U ** 2 + V ** 2)
            all_u.append(U)
            all_v.append(V)
            all_mag.append(mag)
    return XX, YY, all_u, all_v, all_mag


def draw_paths(ax, trajectory, colors, frame_idx, axis_limit, title):
    for p_idx in range(trajectory.shape[1]):
        path = trajectory[: frame_idx + 1, p_idx, :]
        ax.plot(path[:, 0], path[:, 1], color=colors[p_idx], alpha=0.45, linewidth=0.8)
        ax.scatter(path[-1, 0], path[-1, 1], color=colors[p_idx], s=16, edgecolor="black", linewidth=0.25)
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12)
    ax.grid(True, linewidth=0.3, alpha=0.4)


def save_keyframes(trajectory, t_values, colors, args, dist_name, XX, YY, all_u, all_v, all_mag):
    KEYFRAME_DIR.mkdir(parents=True, exist_ok=True)
    key_indices = [0, len(trajectory) // 2, len(trajectory) - 1]
    vmax = float(np.percentile(np.stack(all_mag), 99))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.7), dpi=110)
    for ax, idx in zip(axes, key_indices):
        mag = all_mag[idx]
        U = all_u[idx] / (mag + 1e-8)
        V = all_v[idx] / (mag + 1e-8)
        q = ax.quiver(XX, YY, U, V, mag, cmap="viridis", scale=25, width=0.004, clim=(0, vmax))
        draw_paths(ax, trajectory, colors, idx, args.axis_limit, f"t = {t_values[idx]:.2f}")

    fig.colorbar(q, ax=axes, label="|velocity|", shrink=0.8)
    fig.suptitle(f"Flow Matching transport - {dist_name}", fontsize=13)
    fig.tight_layout()
    out_path = KEYFRAME_DIR / f"flow_matching_animation_{dist_name}_keyframes.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved keyframes to {out_path.relative_to(PROJECT_ROOT)}")


def save_reverse_comparison(args, fm_trajectory, colors):
    score_path = Path(args.score_checkpoint)
    if not score_path.exists():
        print(f"Skipping reverse-SDE comparison: missing {score_path}")
        return

    device = default_device()
    wrapper, process, config = load_score_wrapper(score_path, device=device)
    _, reverse_trajectory = sample_reverse_sde(
        wrapper,
        process,
        n_samples=args.n_particles,
        n_steps=args.steps,
        seed=args.seed,
        device=device,
        return_trajectory=True,
        record_every=args.record_every,
    )

    KEYFRAME_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=120)
    draw_paths(axes[0], reverse_trajectory, colors, len(reverse_trajectory) - 1, args.axis_limit, "reverse-SDE paths")
    draw_paths(axes[1], fm_trajectory, colors, len(fm_trajectory) - 1, args.axis_limit, "Flow Matching paths")
    fig.suptitle(f"Path comparison - {config['distribution']}", fontsize=13)
    fig.tight_layout()
    out_path = KEYFRAME_DIR / f"flow_matching_vs_reverse_sde_{config['distribution']}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison to {out_path.relative_to(PROJECT_ROOT)}")


def main():
    args = parse_args()
    device = default_device()
    model, config = load_velocity_model(args.checkpoint, device=device)
    dist_name = config["distribution"]
    base_std = config.get("base_std", 1.0)

    _, trajectory = sample_flow_matching_ode(
        model,
        n_samples=args.n_particles,
        n_steps=args.steps,
        seed=args.seed,
        method=args.method,
        base_std=base_std,
        device=device,
        return_trajectory=True,
        record_every=args.record_every,
    )
    t_values = np.linspace(0.0, 1.0, len(trajectory))
    XX, YY, all_u, all_v, all_mag = velocity_grid(model, device, args.grid_size, args.grid_limit, t_values)
    vmax = float(np.percentile(np.stack(all_mag), 99))
    colors = plt.cm.hsv(np.linspace(0, 1, args.n_particles, endpoint=False))

    save_keyframes(trajectory, t_values, colors, args, dist_name, XX, YY, all_u, all_v, all_mag)
    save_reverse_comparison(args, trajectory, colors)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
    mag0 = all_mag[0]
    q = ax.quiver(
        XX,
        YY,
        all_u[0] / (mag0 + 1e-8),
        all_v[0] / (mag0 + 1e-8),
        mag0,
        cmap="viridis",
        scale=25,
        width=0.004,
        clim=(0, vmax),
    )
    lines = [ax.plot([], [], color=colors[p], alpha=0.45, linewidth=0.8)[0] for p in range(args.n_particles)]
    points = ax.scatter(trajectory[0, :, 0], trajectory[0, :, 1], color=colors, s=16, edgecolor="black", linewidth=0.25)
    ax.set_xlim(-args.axis_limit, args.axis_limit)
    ax.set_ylim(-args.axis_limit, args.axis_limit)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    title = ax.set_title("")

    def update(frame_idx):
        mag = all_mag[frame_idx]
        q.set_UVC(all_u[frame_idx] / (mag + 1e-8), all_v[frame_idx] / (mag + 1e-8), mag)
        for p_idx in range(args.n_particles):
            path = trajectory[: frame_idx + 1, p_idx, :]
            lines[p_idx].set_data(path[:, 0], path[:, 1])
        points.set_offsets(trajectory[frame_idx])
        title.set_text(f"Flow Matching - {dist_name} - t={t_values[frame_idx]:.2f}")
        return lines + [q, points, title]

    anim = animation.FuncAnimation(fig, update, frames=len(trajectory), blit=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "flow_matching.mp4"
    anim.save(out_path, writer="ffmpeg", fps=args.fps, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
